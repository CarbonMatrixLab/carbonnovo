import torch
from torch.nn import functional as F

from carbonmatrix.model.utils import batched_select

def interp(x, xp, fp, batch_dims=0):
    # x,  (b, L, s); each sample have different t
    # xp, (b, N)
    # fp, (b, N)
    x_shape, xp_shape, fp_shape = list(x.shape), list(xp.shape), list(fp.shape)
    assert (xp_shape == fp_shape and x_shape[:batch_dims] == xp_shape[:batch_dims] and len(xp_shape) == batch_dims + 1)
    
    N = xp_shape[-1]
    
    slope = (fp[...,1:] - fp[...,:-1]) / (xp[...,1:] - xp[...,:-1])
    slope = F.pad(slope, (0, 1))

    x = torch.reshape(x, x_shape[:batch_dims] + [-1])
    
    indices = torch.sum(torch.gt(x[...,None], xp[...,None,:]), dim=-1) - 1
    indices = torch.clip(indices, min=0)
    
    x_0 = batched_select(xp, indices, batch_dims = batch_dims)
    y_0 = batched_select(fp, indices, batch_dims = batch_dims)
    a = batched_select(slope, indices, batch_dims = batch_dims)
    
    v = a * torch.clip(x - x_0, min = 0.) + y_0
    
    v = torch.reshape(v, x_shape)

    return v
