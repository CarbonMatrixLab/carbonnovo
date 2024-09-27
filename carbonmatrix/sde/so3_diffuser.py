import os
import logging

import numpy as np
import torch

from carbonmatrix.model.utils import batched_select, l2_normalize
from carbonmatrix.sde.utils import interp
from carbonmatrix.sde.igso3 import calc_so3_score
from carbonmatrix.model import quat_affine

class SO3Diffuser:
    def __init__(self, so3_conf):
        self.schedule = so3_conf.schedule

        self.min_sigma = so3_conf.min_sigma
        self.max_sigma = so3_conf.max_sigma

        self.num_sigma = so3_conf.num_sigma
        self.num_omega = so3_conf.num_omega 

        igso3_density_data = np.load(so3_conf.igso3_density_data)

        self._cdf = torch.tensor(igso3_density_data['cdf'], dtype=torch.float32)
        self._score_norms = torch.tensor(igso3_density_data['score_norms'], dtype=torch.float32)
        self._score_scaling = torch.tensor(igso3_density_data['score_scaling'], dtype=torch.float32)

        self.discrete_sigma = self.sigma(torch.linspace(0.0, 1.0, self.num_sigma))
        self.discrete_omega = torch.linspace(0, np.pi, so3_conf.num_omega+1)[1:]

    def sigma_idx(self, sigma):
        indices = torch.sum(sigma[...,None] >= self.discrete_sigma.to(device=sigma.device), axis=-1) - 1
        return indices

    def sigma(self, t): #t, tensor (L,)
        if self.schedule == 'logarithmic':
            return torch.log(t * torch.exp(torch.tensor(self.max_sigma)) + (1-t) * torch.exp(torch.tensor(self.min_sigma)))
        else:
            raise ValueError(f'Unrecognize schedule {self.schedule}')

    def diffusion_coef(self, t): #t, tensor(L,)
        """Compute diffusion coefficient (g_t)."""
        if self.schedule == 'logarithmic':
            sigma_t = self.sigma(t)
            g_t = torch.sqrt(
                2 * (torch.exp(torch.tensor(self.max_sigma)) - torch.exp(torch.tensor(self.min_sigma))) * sigma_t / torch.exp(sigma_t)
                )
        else:
            raise ValueError(f'Unrecognize schedule {self.schedule}')
        return g_t

    def t_to_idx(self, t):
        s = self.sigma(t)
        return self.sigma_idx(s)
    
    def _sample_igso3(self, t, samples_shape):
        """Uses the inverse cdf to sample an angle of rotation from IGSO(3).

        Args:
            t: continuous time in [0, 1].
            samples_shape: number of samples to draw.

        Returns:
            samples_shape of angles of rotation.
        """
        bs, device = t.shape[0], t.device
        
        x = torch.rand(samples_shape, device = device)
        
        t_idx = self.t_to_idx(t) 

        cdf = batched_select(self._cdf.to(device = device), t_idx)
        omega = torch.tile(self.discrete_omega.to(device = device), (bs, 1))

        return interp(x, cdf, omega, batch_dims=1)

    def sample(self, t, samples_shape):
        device = t.device

        x = torch.randn(list(samples_shape) + [3], device = device)
        x = l2_normalize(x)
        
        omega = self._sample_igso3(t, samples_shape)
        return x * omega[..., None]

    def sample_ref(self, t, samples_shape):
        sampled_axis_angle = self.sample(t, samples_shape=samples_shape)
        sampled_quat = quat_affine.axis_angle_to_quaternion(sampled_axis_angle)
        return sampled_quat

    def score(self, axis_angle, t, eps = 1e-10):
        bs, device = t.shape[0], t.device
       
        #(b, l)
        omega = torch.linalg.norm(axis_angle, dim=-1) + eps
        
        t_idx = self.t_to_idx(t)
        sigma = batched_select(self.discrete_sigma.to(device=device), t_idx)
       
        omega_scores_t = calc_so3_score(omega, sigma)
        
        score_t = omega_scores_t[..., None] * axis_angle / (omega[..., None] + eps)
        
        return score_t

    def score_scaling(self, t):
        t_idx = self.t_to_idx(t)
        return batched_select(self._score_scaling.to(device=t.device), t_idx)

    def forward_marginal(self, quat_0, t):
        samples_shape = list(quat_0.shape)[:-1]
        
        sampled_axis_angle = self.sample(t, samples_shape = samples_shape)
        sampled_quat = quat_affine.axis_angle_to_quaternion(sampled_axis_angle)
        quat_t = quat_affine.quat_multiply(quat_0, sampled_quat) 
        
        score_t = self.score(sampled_axis_angle, t)

        return quat_t, score_t

    def reverse(self, quat_t, score_t, mask, t: torch.Tensor, dt: float, noise_scale: float=1.0,):
        g_t = self.diffusion_coef(t)
        z = noise_scale * torch.normal(mean=0., std=1., size=score_t.shape).to(device=quat_t.device)
        perturb = (g_t ** 2) * score_t * dt + g_t * np.sqrt(dt) * z

        if mask is not None:
            perturb = perturb * mask[..., None]

        # Right multiply.
        perturb_quat = quat_affine.axis_angle_to_quaternion(perturb)
        quat_t_1 = quat_affine.quat_multiply(quat_t, perturb_quat)
        
        return quat_t_1
