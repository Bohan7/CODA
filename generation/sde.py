"""Forward SDEs for node features X and adjacency A (VP and VE), following GDSS / Song et al. (2021)."""
import numpy as np
import torch


class SDE:
    """Base class. `N` is the number of discretisation steps, time runs over [0, T] with T = 1."""

    T = 1

    def __init__(self, N):
        self.N = N

    def sde(self, x, t):
        """Drift f(x, t) and diffusion g(t) of the forward SDE."""
        raise NotImplementedError

    def marginal_prob(self, x, t):
        """Mean and std of the perturbation kernel p_t(x_t | x_0)."""
        raise NotImplementedError

    def discretize(self, x, t):
        """One-step discretisation x_{i+1} = x_i + f_i(x_i) + G_i z_i."""
        raise NotImplementedError

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_sampling_sym(self, shape):
        """Symmetric prior sample with zero diagonal (for adjacency matrices)."""
        x = torch.randn(*shape).triu(1)
        return x + x.transpose(-1, -2)

    def reverse(self):
        """Reverse-time SDE: drift f - g^2 * score, same diffusion g."""
        N, sde_fn, discretize_fn = self.N, self.sde, self.discretize

        class RSDE:
            def __init__(self):
                self.N = N

            def sde(self, z, t, score):
                drift, diffusion = sde_fn(z, t)
                drift = drift - diffusion[:, None, None] ** 2 * score
                return drift, diffusion

            def discretize(self, z, t, score):
                f, G = discretize_fn(z, t)
                rev_f = f - G[:, None, None] ** 2 * score
                return rev_f, G

        return RSDE()


class VPSDE(SDE):
    """Variance-preserving SDE with a linear beta schedule on [beta_min, beta_max]."""

    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def discretize(self, x, t):
        """DDPM discretisation."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        f = torch.sqrt(alpha)[:, None, None] * x - x
        G = torch.sqrt(beta)
        return f, G


class VESDE(SDE):
    """Variance-exploding SDE with a geometric sigma schedule on [sigma_min, sigma_max]."""

    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(sigma_min), np.log(sigma_max), N))

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                    device=t.device))
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        return x, std

    def discretize(self, x, t):
        """SMLD (NCSN) discretisation."""
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                     self.discrete_sigmas.to(t.device)[timestep - 1])
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G


def load_sde(cfg):
    """Build an SDE from a config node with fields `type`, `beta_min`, `beta_max`, `num_scales`."""
    if cfg.type == 'VP':
        return VPSDE(beta_min=cfg.beta_min, beta_max=cfg.beta_max, N=cfg.num_scales)
    if cfg.type == 'VE':
        return VESDE(sigma_min=cfg.beta_min, sigma_max=cfg.beta_max, N=cfg.num_scales)
    raise NotImplementedError(f'SDE type {cfg.type} is not supported.')
