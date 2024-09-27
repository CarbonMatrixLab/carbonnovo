"""R^3 diffusion methods."""
import numpy as np
from scipy.special import gamma

import torch

class R3Diffuser:
    """VP-SDE diffuser class for translations."""

    def __init__(self, r3_conf):
        """
        Args:
            min_b: starting value in variance schedule.
            max_b: ending value in variance schedule.
        """
        self._r3_conf = r3_conf
        self.min_b = r3_conf.min_b
        self.max_b = r3_conf.max_b

    def _scale(self, x):
        return x * self._r3_conf.coordinate_scaling

    def _unscale(self, x):
        return x / self._r3_conf.coordinate_scaling

    def b_t(self, t : torch.Tensor): #t: (bs,)
        return self.min_b + t * (self.max_b - self.min_b)

    def diffusion_coef(self, t : torch.Tensor):
        """Time-dependent diffusion coefficient."""
        return torch.sqrt(self.b_t(t))

    def drift_coef(self, x, t : torch.Tensor): #t: (bs,)
        """Time-dependent drift coefficient."""
        return -1/2 * self.b_t(t)[:,None,None] * x

    def sample_ref(self, t, samples_shape):
        device = t.device
        mean = torch.full(samples_shape + [3], 0.0, device=device)
        std = torch.full(samples_shape + [3], 1.0, device=device)

        return torch.normal(mean = mean, std = std)

    def marginal_b_t(self, t:torch.Tensor):
        return t * self.min_b + (1/2)*(t**2)*(self.max_b-self.min_b)

    def calc_trans_0(self, score_t, x_t, t, use_torch=True):
        beta_t = self.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        exp_fn = torch.exp if use_torch else np.exp
        cond_var = 1 - exp_fn(-beta_t)
        return (score_t * cond_var + x_t) / exp_fn(-1/2*beta_t)

    def forward(self, x_t_1: np.ndarray, t: float, num_t: int):
        """Samples marginal p(x(t) | x(t-1)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        x_t_1 = self._scale(x_t_1)
        b_t = torch.tensor(self.marginal_b_t(t) / num_t).to(x_t_1.device)
        z_t_1 = torch.tensor(np.random.normal(size=x_t_1.shape)).to(x_t_1.device)
        x_t = torch.sqrt(1 - b_t) * x_t_1 + torch.sqrt(b_t) * z_t_1
        return x_t

    def distribution(self, x_t, score_t, t, mask, dt):
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        std = g_t * np.sqrt(dt)
        mu = x_t - (f_t - g_t**2 * score_t) * dt
        if mask is not None:
            mu *= mask[..., None]
        return mu, std

    def forward_marginal(self, x_0, t):
        """Samples marginal p(x(t) | x(0)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """

        x_0 = self._scale(x_0)

        log_mean_coeff = -0.5 * self.marginal_b_t(t)

        cast_shape = [log_mean_coeff.shape[0]] + [1] * (x_0.ndim - 1)
        log_mean_coeff = torch.reshape(log_mean_coeff, cast_shape)

        mean = torch.exp(log_mean_coeff) * x_0
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))

        x_t = torch.normal(mean = mean, std = std)

        score_t = self.score(x_t, x_0, t)

        x_t = self._unscale(x_t)

        return x_t, score_t

    def score_scaling(self, t):
        return 1 / torch.sqrt(self.conditional_var(t))

    def reverse(self, x_t, score_t, mask, t: torch.Tensor, dt: float, center: bool=True, noise_scale: float=1.0,):
        """Simulates the reverse SDE for 1 step

        Args:
            x_t: [..., 3] current positions at time t in angstroms.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
        Returns:
            [..., 3] positions at next step t-1.
        """
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)[:,None,None]
        f_t = self.drift_coef(x_t, t)
        z = noise_scale * torch.normal(mean=0., std=1., size=score_t.shape).to(device=x_t.device)
        perturb = (f_t - g_t**2 * score_t) * dt + g_t * np.sqrt(dt) * z

        x_t_1 = x_t - perturb
        if center:
            com = torch.sum(x_t_1, dim=-2) / torch.sum(mask, dim=-1, keepdims=True)
            x_t_1 = x_t_1 - com[..., None, :]
        x_t_1 = self._unscale(x_t_1)
        return x_t_1

    def conditional_var(self, t:torch.Tensor):
        """Conditional variance of p(xt|x0).

        Var[x_t|x_0] = conditional_var(t)*I

        """
        return 1 - torch.exp(-self.marginal_b_t(t))

    def score(self, x_t, x_0, t:torch.Tensor, scale=True):
        if scale:
            x_t = self._scale(x_t)
            x_0 = self._scale(x_0)

        t = t[:,None,None]

        return -(x_t - torch.exp(-1/2*self.marginal_b_t(t)) * x_0) / self.conditional_var(t)
