import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import LayerNorm as LayerNorm
from einops import rearrange

from carbonmatrix.common import residue_constants

def get_lora_config(config):
    return config.get('lora_config', {})

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, init,
            bias=True, device=None, dtype=None,
            lora_r = 0, lora_alpha = 1., lora_dropout = 0.):
        self.lora_r = lora_r
        self.init = init
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

        if self.lora_r > 0:
            factory_kwargs = {'device': device, 'dtype': dtype}
            self.lora_A = nn.Parameter(torch.empty((in_features, lora_r), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((lora_r, out_features), **factory_kwargs))
            self.scaling = lora_alpha / self.lora_r

            if lora_dropout > 0.:
                self.lora_dropout = nn.Dropout(p=lora_dropout)
            else:
                self.lora_dropout = lambda x: x

            self._reset_lora_parameters()

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

    def reset_parameters(self, ):
        init = self.init
        assert init in ['gate', 'final', 'attn', 'relu', 'linear']

        if init in ['gate', 'final']:
            nn.init.constant_(self.weight, 0.)
        elif init == 'attn':
            # GlorotUniform
            torch.nn.init.xavier_uniform_(self.weight)
        elif init in ['relu', 'linear']:
            # Relu, He
            # linear, Le cun
            distribution_stddev = 0.87962566103423978
            scale = 2. if init == 'relu' else 1.
            stddev = np.sqrt(scale / self.in_features) / distribution_stddev
            nn.init.trunc_normal_(self.weight, mean=0., std=stddev)
        else:
            raise NotImplementedError(f'{init} not Implemented')

        if self.bias is not None:
            if init == 'gate':
                nn.init.constant_(self.bias, 1.)
            else:
                nn.init.constant_(self.bias, 0.)

    def _reset_lora_parameters(self, ):
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5.))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        if self.lora_r > 0:
            result = F.linear(x, self.weight, self.bias)
            result = result + (self.lora_dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
            return result
        else:
            return F.linear(x, self.weight, self.bias)

def apply_dropout(tensor, rate, is_training, broadcast_dim=None):
    if is_training and rate > 0.0:
        if broadcast_dim is not None:
            shape = list(tensor.shape)
            shape[broadcast_dim] = 1
            with torch.no_grad():
                scale = 1. / (1. - rate)
                keep_rate = torch.full(shape, 1. - rate, dtype=tensor.dtype, device=tensor.device)
                keep = torch.bernoulli(keep_rate)
            return scale * keep * tensor
        else:
            return F.dropout(tensor, rate)
    else:
        return tensor

def pseudo_beta_fn_v2(aatype, all_atom_positions, all_atom_masks=None):
    """all_atom_positions is in atom37 format"""

    n_idx = residue_constants.atom_order['N']
    ca_idx = residue_constants.atom_order['CA']
    c_idx = residue_constants.atom_order['C']

    N = all_atom_positions[..., n_idx, :]
    CA = all_atom_positions[..., ca_idx, :]
    C = all_atom_positions[..., c_idx, :]

    b = CA - N
    c = C - CA
    a = torch.cross(b, c, dim=-1)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

    if all_atom_masks is not None:
        CB_mask = torch.all(
                torch.stack([all_atom_masks[...,n_idx], all_atom_masks[...,ca_idx], all_atom_masks[...,c_idx]], dim=-1), dim=-1)
        return CB, CB_mask

    return CB

def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    """Create pseudo beta features."""

    is_gly = torch.eq(aatype, residue_constants.restype_order['G'])
    ca_idx = residue_constants.atom_order['CA']
    cb_idx = residue_constants.atom_order['CB']

    pseudo_beta = torch.where(
        #torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
        is_gly[...,None],
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :])

    if all_atom_masks is not None:
        pseudo_beta_mask = torch.where(
            is_gly,
            all_atom_masks[..., ca_idx].to(dtype=torch.float32),
            all_atom_masks[..., cb_idx].to(dtype=torch.float32))
        return pseudo_beta, pseudo_beta_mask

    return pseudo_beta

def dgram_from_positions(positions, num_bins, min_bin, max_bin):
    breaks = torch.linspace(min_bin, max_bin, steps=num_bins-1, device=positions.device)
    sq_breaks = torch.square(breaks)

    dist2 = torch.sum(
        torch.square(
            rearrange(positions, 'b l c -> b l () c') -
            rearrange(positions, 'b l c -> b () l c')),
        dim=-1,
        keepdims=True)

    true_bins = torch.sum(dist2 > sq_breaks, axis=-1).long()

    return true_bins
