"""Predictor-corrector sampler for the CODA reverse-time SDE.

At every reverse step the unconditional score s_theta(G_t) is combined with the gradient of the
time-dependent classifier phi_t and tempered by the exploration strength lambda:

    score_CODA = (1 - sqrt(lambda)) * [ s_theta(G_t) + alpha_t * grad_{G_t} log p_t(y | G_t) ]

with alpha_t = r * 0.1^t * ||s_theta|| / ||grad log p_t(y | G_t)|| (entry-wise L1 norms, per graph).
lambda = 0 recovers plain classifier-guided class-conditional generation (the GDSS-CC baseline).
"""
import math

import numpy as np
import torch
from tqdm import trange

from sde import VPSDE
from utils.graph_utils import gen_noise, mask_adjs, mask_x


# ---------------------------------------------------------------- score functions
def get_score_fn(sde_x, sde_adj, model, train=False):
    """Wrap the score network: VP models predict the noise, VE models predict the score directly."""
    if not train:
        model.eval()

    def compute_score(pred, sde, t):
        if isinstance(sde, VPSDE):
            std = sde.marginal_prob(torch.zeros_like(pred), t)[1]
            return -pred / std[:, None, None]
        return pred

    def score_fn(x, adj, flags, t):
        pred_x, pred_adj = model(x, adj, t.unsqueeze(-1), flags)
        return compute_score(pred_x, sde_x, t), compute_score(pred_adj, sde_adj, t)

    return score_fn


def get_coda_score_fn(score_fn, classifier, labels, weight_x, weight_adj, lam):
    """Tempered, classifier-guided score of the CODA reverse SDE.

    The OOD surrogate p(E_ood = lambda | G, y) ∝ p(G, y)^(-sqrt(lambda)) scales the conditional score by 1 - sqrt(lambda).
    """
    factor = 1 - math.sqrt(lam)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    def guided_score(x, adj, flags, t):
        score_x, score_adj = score_fn(x, adj, flags, t)

        grad_x = torch.zeros_like(score_x)
        grad_adj = torch.zeros_like(score_adj)
        if weight_x or weight_adj:
            # gradient of log p_t(y | G_t) with respect to the noisy graph
            classifier.eval()
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                adj_in = adj.detach().requires_grad_(True)
                _, _, logits = classifier(x_in, adj_in, t.unsqueeze(-1), flags)
                log_prob = -criterion(logits, labels.to(adj.device))
                x_grad, adj_grad = torch.autograd.grad(log_prob, [x_in, adj_in])
            batch = adj.shape[0]
            schedule = 0.1 ** t[0].item()
            if weight_adj:
                grad_adj = mask_adjs(adj_grad, flags)
                ratio = score_adj.view(batch, -1).norm(p=1, dim=-1) / (grad_adj.view(batch, -1).norm(p=1, dim=-1) + 1e-18)
                grad_adj *= (weight_adj * schedule) * ratio[:, None, None]
            if weight_x:
                grad_x = mask_x(x_grad, flags)
                ratio = score_x.view(batch, -1).norm(p=1, dim=-1) / (grad_x.view(batch, -1).norm(p=1, dim=-1) + 1e-18)
                grad_x *= (weight_x * schedule) * ratio[:, None, None]

        score_x *= factor
        score_adj *= factor
        grad_x *= factor
        grad_adj *= factor
        return score_x + grad_x, score_adj + grad_adj

    return guided_score


# ---------------------------------------------------------------- predictors
class EulerMaruyamaPredictor:
    def __init__(self, sde_x, sde_adj, score_fn):
        self.rsde_x = sde_x.reverse()
        self.rsde_adj = sde_adj.reverse()
        self.score_fn = score_fn

    def update_fn(self, x, adj, flags, t):
        dt = -1. / self.rsde_x.N
        score_x, score_adj = self.score_fn(x, adj, flags, t)

        drift_x, diffusion_x = self.rsde_x.sde(x, t, score_x)
        x_mean = x + drift_x * dt
        x = x_mean + diffusion_x[:, None, None] * np.sqrt(-dt) * gen_noise(x, flags, sym=False)

        drift_adj, diffusion_adj = self.rsde_adj.sde(adj, t, score_adj)
        adj_mean = adj + drift_adj * dt
        adj = adj_mean + diffusion_adj[:, None, None] * np.sqrt(-dt) * gen_noise(adj, flags, sym=True)
        return x, adj, x_mean, adj_mean


class ReverseDiffusionPredictor:
    def __init__(self, sde_x, sde_adj, score_fn):
        self.rsde_x = sde_x.reverse()
        self.rsde_adj = sde_adj.reverse()
        self.score_fn = score_fn

    def update_fn(self, x, adj, flags, t):
        score_x, score_adj = self.score_fn(x, adj, flags, t)

        f_x, G_x = self.rsde_x.discretize(x, t, score_x)
        z = gen_noise(x, flags, sym=False)
        x_mean = x - f_x
        x = x_mean + G_x[:, None, None] * z

        f_adj, G_adj = self.rsde_adj.discretize(adj, t, score_adj)
        z = gen_noise(adj, flags, sym=True)
        adj_mean = adj - f_adj
        adj = adj_mean + G_adj[:, None, None] * z
        return x, adj, x_mean, adj_mean


# ---------------------------------------------------------------- correctors
class NoneCorrector:
    def update_fn(self, x, adj, flags, t):
        return x, adj, x, adj


class LangevinCorrector:
    """One Langevin step per reverse step, with the step size set by the signal-to-noise ratio `snr`."""

    def __init__(self, sde_x, sde_adj, score_fn, snr, scale_eps):
        self.sde_x, self.sde_adj, self.score_fn = sde_x, sde_adj, score_fn
        self.snr, self.scale_eps = snr, scale_eps

    def _langevin(self, z, score, sde, flags, t, sym):
        if isinstance(sde, VPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)
        noise = gen_noise(z, flags, sym=sym)
        grad_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        z_mean = z + step_size[:, None, None] * score
        z = z_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * self.scale_eps
        return z, z_mean

    def update_fn(self, x, adj, flags, t):
        score_x, score_adj = self.score_fn(x, adj, flags, t)
        x, x_mean = self._langevin(x, score_x, self.sde_x, flags, t, sym=False)
        adj, adj_mean = self._langevin(adj, score_adj, self.sde_adj, flags, t, sym=True)
        return x, adj, x_mean, adj_mean


# ---------------------------------------------------------------- sampler
PREDICTORS = {'Euler': EulerMaruyamaPredictor, 'Reverse': ReverseDiffusionPredictor}


def get_coda_sampler(sde_x, sde_adj, shape_x, shape_adj, predictor='Euler', corrector='None',
                     snr=0.1, scale_eps=1.0, weight_x=0., weight_adj=0.,
                     lam=0., denoise=True, eps=1e-3, device='cuda'):
    """Return `sampler(model, classifier, flags, labels) -> (x, adj)` running the full reverse process."""
    if predictor not in PREDICTORS:
        raise ValueError(f'Unknown predictor: {predictor} (choose from {list(PREDICTORS)})')
    corrector = corrector or 'None'
    if corrector not in ('Langevin', 'None'):
        raise ValueError(f'Unknown corrector: {corrector} (choose from Langevin, None)')

    def sampler(model, classifier, flags, labels):
        score_fn = get_score_fn(sde_x, sde_adj, model)
        score_fn = get_coda_score_fn(score_fn, classifier, labels, weight_x, weight_adj, lam)

        predictor_obj = PREDICTORS[predictor](sde_x, sde_adj, score_fn)
        if corrector == 'Langevin':
            corrector_obj = LangevinCorrector(sde_x, sde_adj, score_fn, snr, scale_eps)
        else:
            corrector_obj = NoneCorrector()

        with torch.no_grad():
            x = mask_x(sde_x.prior_sampling(shape_x).to(device), flags)
            adj = mask_adjs(sde_adj.prior_sampling_sym(shape_adj).to(device), flags)
            timesteps = torch.linspace(sde_adj.T, eps, sde_adj.N, device=device)

            for i in trange(sde_adj.N, desc='[Sampling]', leave=False):
                vec_t = torch.ones(shape_adj[0], device=device) * timesteps[i]
                x, adj, x_mean, adj_mean = corrector_obj.update_fn(x, adj, flags, vec_t)
                x, adj, x_mean, adj_mean = predictor_obj.update_fn(x, adj, flags, vec_t)

        return (x_mean if denoise else x), (adj_mean if denoise else adj)

    return sampler
