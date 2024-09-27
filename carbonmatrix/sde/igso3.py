import torch

from einops import rearrange

def igso3_expansion(omega, sigma, L=1000):
    ''' 
    omega: (batch_size, num_res). the same dimension with t
    sigma: (batch_size,). the same dimension with t

    return:
       expansions: (batch_size, num_res)
    '''
    assert (omega.dim() == 2)
    
    ls = torch.arange(L, device=omega.device)
    
    omega = rearrange(omega, 'b n -> b n ()')
    sigma = rearrange(sigma, 'b -> b () ()')
    ls = rearrange(ls, 'l -> () () l')
    
    p = (2*ls + 1) * torch.exp(-ls*(ls+1)*sigma**2/2.) * torch.sin(omega*(ls+0.5)) / torch.sin(omega/2.)

    return torch.sum(p, dim=-1)

# score of density over SO(3)
def igso3_score(expans, omega, sigma, L=1000):
    ''' 
    expans: (batch_size, num_res). igso3 expansions
    omega: (batch_size, num_res). the same dimension with t
    sigma: (batch_size,). the same dimension with t

    return:
       expansions: (batch_size, num_res)
    '''
    ls = torch.arange(L, device=omega.device)
    
    omega = rearrange(omega, 'b n -> b n ()')
    sigma = rearrange(sigma, 'b -> b () ()')
    ls = rearrange(ls, 'l -> () () l')
    
    hi = torch.sin(omega * (ls + 1 / 2))
    dhi = (ls + 1 / 2) * torch.cos(omega * (ls + 1 / 2))
    lo = torch.sin(omega / 2)
    dlo = 1 / 2 * torch.cos(omega / 2)
    dSigma = (2 * ls + 1) * torch.exp(-ls * (ls + 1) * sigma**2/2) * (lo * dhi - hi * dlo) / lo ** 2
    
    dSigma = torch.sum(dSigma, dim=-1)
    
    return dSigma / (expans + 1e-4)

def calc_so3_score(omega, sigma, L=1000):
    ''' 
    omega: (batch_size, num_res). the same dimension with t
    sigma: (batch_size,). the same dimension with t

    return:
       expansions: (batch_size, num_res)
    '''
    assert (omega.dim() == 2 and sigma.dim() == 1)
    
    
    ls = torch.arange(L, device=omega.device)
    
    omega = rearrange(omega, 'b n -> b n ()')
    sigma = rearrange(sigma, 'b -> b () ()')
    ls = rearrange(ls, 'l -> () () l')
    
    p = (2.*ls + 1.) * torch.exp(-ls*(ls+1)*sigma**2 * 0.5) * torch.sin(omega*(ls+0.5)) / torch.sin(omega * 0.5)
    # (batch_size, num_res)
    expans = torch.sum(p, dim=-1)
    
    hi = torch.sin(omega * (ls + 0.5))
    dhi = (ls + 0.5) * torch.cos(omega * (ls + 0.5))
    lo = torch.sin(omega * 0.5)
    dlo = 0.5 * torch.cos(omega * 0.5)
    dSigma = (2. * ls + 1.) * torch.exp(-ls * (ls + 1.) * sigma**2 * 0.5) * (lo * dhi - hi * dlo) / lo ** 2
    
    dSigma = torch.sum(dSigma, dim=-1)
    
    return dSigma / (expans + 1e-8)
