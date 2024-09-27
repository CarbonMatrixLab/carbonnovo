import torch
from torch import nn
from einops import rearrange

from carbonmatrix.common import residue_constants
from carbonmatrix.model import folding
from carbonmatrix.model.inverseformer import InverseFormer
from carbonmatrix.model.common_modules import Linear, LayerNorm
from carbonmatrix.model.common_modules import get_lora_config
from carbonmatrix.model.head_factory import registry_head

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

@registry_head(name='distogram')
class DistogramHead(nn.Module):
    """Head to predict a distogram.
    """
    def __init__(self, config, num_in_pair_channel):
        super().__init__()

        c = config
        lora_config = get_lora_config(c)

        self.breaks = torch.linspace(c.first_break, c.last_break, steps=c.num_bins-1)
        self.proj = Linear(num_in_pair_channel, c.num_bins, init='final', **lora_config)

        self.config = config

    def forward(self, headers, representations, batch):
        x = representations['pair']
        x = self.proj(x)
        logits = (x + rearrange(x, 'b i j c -> b j i c')) * 0.5
        breaks = self.breaks.to(logits.device)
        return dict(logits=logits, breaks=breaks)

@registry_head(name='structure_module')
class FoldingHead(nn.Module):
    """Head to predict 3d struct.
    """
    def __init__(self, config, num_in_seq_channel, num_in_pair_channel):
        super().__init__()
        self.struct_module = folding.StructureModule(config, num_in_seq_channel, num_in_pair_channel)

        self.config = config

    def forward(self, headers, representations, batch):
        return self.struct_module(representations, batch)


class PredictedLDDTHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config

        dim = c.num_channel

        self.net = nn.Sequential(
                LayerNorm(c.structure_module_num_channel),
                Linear(c.structure_module_num_channel, dim, init='relu', bias=True),
                nn.ReLU(),
                Linear(dim, dim, init='relu', bias=True),
                nn.ReLU(),
                Linear(dim, c.num_bins, init='final', bias=True))
        self.config = config

    def forward(self, headers, representations, batch):
        assert 'structure_module' in headers

        act = headers['structure_module']['representations']['structure_module']

        return dict(logits=self.net(act))


@registry_head(name='sequence')
class SequenceHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config
        self.config = c
        ###seq part
        seq_channel = 384
        self.seq_norm = LayerNorm(seq_channel)
        self.seq_proj = Linear(384, len(residue_constants.restypes_with_x), init='final')
        self.pre_seq_proj = Linear(384, 384, init='final', bias=True)

        ###pair part
        pair_channel = 128
        num_token = len(residue_constants.restypes_with_x)
        self.pair_norm = LayerNorm(pair_channel)
        self.pair_proj = Linear(pair_channel, num_token * num_token, init='final')
        self.num_token = num_token

        self.inverseformer = InverseFormer(config)

    def forward(self, headers, representations, batch):
        # seq part
        si = headers['structure_module']['representations']['structure_module']

        # feature
        _, pred_trans = headers['structure_module']['traj'][-1]
        ca_one_hot = dgram_from_positions(pred_trans, 20, 2.0, 12.0)
        ca_one_hot = torch.nn.functional.one_hot(ca_one_hot, num_classes=20).to(torch.float)

        pad_mask = batch['mask']
        pair_pad_mask = rearrange(pad_mask, 'b i -> b () i') * rearrange(pad_mask, 'b j -> b j ()')
        shape = pair_pad_mask.shape
        diag_matrix = torch.eye(shape[2])
        diag_matrix = diag_matrix.unsqueeze(0)
        mrf_mask = torch.logical_not(diag_matrix).to(pad_mask.device)

        ca_dist = (rearrange(pred_trans, 'b i j -> b i () j') - rearrange(pred_trans, 'b i j -> b () i j')) ** 2
        ca_dist = ca_dist.sum(dim=-1)
        dist_mask_ = ca_dist < 144
        dist_mask = dist_mask_.detach()
        dist_mask.to(pad_mask.device)

        total_mask = dist_mask * pair_pad_mask * mrf_mask


        seq_act = self.pre_seq_proj(si)

        seq_act, pair_act = self.inverseformer(batch, seq_act, ca_one_hot, total_mask)

        seq_act = self.seq_norm(seq_act)
        seq_logits = self.seq_proj(seq_act)

        aatype = torch.argmax(seq_logits, dim = -1).detach()

        batch.update({'batch_pred_seq': aatype})
        batch.update({'pair_distall_mask': total_mask})
        ##pair part
        zij = self.pair_norm(pair_act)
        pair_logits = self.pair_proj(zij)
        pair_logits = rearrange(pair_logits, 'b i j (a c) -> b i j a c', a = self.num_token)
        pair_logits = rearrange(0.5 * (pair_logits + rearrange(pair_logits, 'b i j a c -> b j i c a')), 'b i j a c -> b i j (a c)')

        return dict(seq_logits=seq_logits, pair_logits=pair_logits)#, aatype=aatype)


