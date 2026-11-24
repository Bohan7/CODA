"""Training objectives: denoising score matching for s_theta and noisy-graph classification for phi_t."""
import torch

from solver import get_score_fn
from utils.graph_utils import gen_noise, mask_adjs, mask_x, node_flags


def perturb(sde_x, sde_adj, x, adj, eps):
    """Sample t ~ U(eps, T) and draw (X_t, A_t) from the perturbation kernels."""
    t = torch.rand(adj.shape[0], device=adj.device) * (sde_adj.T - eps) + eps
    flags = node_flags(adj)

    z_x = gen_noise(x, flags, sym=False)
    mean_x, std_x = sde_x.marginal_prob(x, t)
    perturbed_x = mask_x(mean_x + std_x[:, None, None] * z_x, flags)

    z_adj = gen_noise(adj, flags, sym=True)
    mean_adj, std_adj = sde_adj.marginal_prob(adj, t)
    perturbed_adj = mask_adjs(mean_adj + std_adj[:, None, None] * z_adj, flags)
    return t, flags, (perturbed_x, z_x, std_x), (perturbed_adj, z_adj, std_adj)


def get_score_loss_fn(sde_x, sde_adj, lambda_train=5.0, reduce_mean=False, eps=1e-5):
    """Denoising score matching on X and A; the two terms are combined as loss_x + lambda_train * loss_adj."""
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, x, adj):
        t, flags, (perturbed_x, z_x, std_x), (perturbed_adj, z_adj, std_adj) = perturb(sde_x, sde_adj, x, adj, eps)
        score_x, score_adj = get_score_fn(sde_x, sde_adj, model, train=True)(perturbed_x, perturbed_adj, flags, t)

        losses_x = torch.square(score_x * std_x[:, None, None] + z_x)
        losses_x = reduce_op(losses_x.reshape(losses_x.shape[0], -1), dim=-1)
        losses_adj = torch.square(score_adj * std_adj[:, None, None] + z_adj)
        losses_adj = reduce_op(losses_adj.reshape(losses_adj.shape[0], -1), dim=-1)

        loss_x, loss_adj = losses_x.mean(), losses_adj.mean()
        return loss_x + lambda_train * loss_adj, loss_x, loss_adj

    return loss_fn


def get_classifier_loss_fn(sde_x, sde_adj, eps=1e-5, class_weights=None):
    """Cross-entropy of phi_t(G_t, t) on noisy graphs, i.e. the classifier approximates p_t(y | G_t)."""

    def loss_fn(model, x, adj, labels):
        t, flags, (perturbed_x, _, _), (perturbed_adj, _, _) = perturb(sde_x, sde_adj, x, adj, eps)
        _, _, logits = model(perturbed_x, perturbed_adj, t.unsqueeze(-1), flags)
        weight = None if class_weights is None else class_weights.to(adj.device)
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
        correct = logits.argmax(dim=1).eq(labels).sum().item()
        return loss, correct

    return loss_fn
