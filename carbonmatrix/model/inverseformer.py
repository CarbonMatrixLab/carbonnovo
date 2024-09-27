import functools
from einops import rearrange

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from carbonmatrix.common import residue_constants
from carbonmatrix.model.common_modules import Linear, LayerNorm, apply_dropout
from carbonmatrix.model.baseformer import TriangleMultiplication, TriangleAttention

class InverseFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config
        self.config = config.inverseformer
        self.num_token = len(residue_constants.restypes_with_x)

        self.proj_rel_pos = Linear(self.config.max_relative_feature * 2 + 2 + 1, self.config.pair_channel, init='linear')
        self.proj_pair_dis = Linear(20, self.config.pair_channel, init='linear')
        self.seqformer = Seqformer(c)

    def forward(self, batch, si, dist, total_mask):
        c = self.config
        seq, mask = batch['seq'], batch['mask']
        batch_size, num_residue = seq.shape[:2]
        device = seq.device

        # sequence distance
        seq_pos = torch.arange(num_residue, device=device)

        offset = rearrange(seq_pos, 'l -> () l ()') - rearrange(seq_pos, 'l -> () () l')
        clipped_offset = torch.clamp(offset + c.max_relative_feature, min=0, max=2*c.max_relative_feature)

        chain_id = torch.zeros((batch_size, num_residue), dtype=torch.int32,device=seq.device)
        eq_chain_id = torch.eq(rearrange(chain_id, 'b l -> b () l'), rearrange(chain_id, 'b l -> b l ()'))

        final_offset = torch.where(eq_chain_id, clipped_offset.to(dtype=torch.float32),  torch.tensor(2*c.max_relative_feature+1, dtype=torch.float32, device=device)).to(dtype=torch.long)

        rel_pos = F.one_hot(final_offset, num_classes = 2 * c.max_relative_feature + 2).to(dtype=torch.float32)
        rel_pos = torch.cat([rel_pos, eq_chain_id.to(dtype=rel_pos.dtype)[:,:,:,None]], axis=-1)

        pair_act = self.proj_rel_pos(rel_pos)

        # residue-residue 3D distances
        pair_act = pair_act + self.proj_pair_dis(dist)

        seq_act = si
        seq_act, pair_act = self.seqformer(
                seq_act, pair_act,
                mask=mask, pair_mask=total_mask,
                is_recycling=batch['is_recycling'])

        return seq_act, pair_act

class Seqformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config

        self.blocks = nn.ModuleList([SeqformerIteration(c.inverseformer, c.inverseformer.seq_channel, c.inverseformer.pair_channel) for _ in range(c.inverseformer.seqformer_num_block)])

    def forward(self, seq_act, pair_act, mask, pair_mask=None, is_recycling=True):
        for block in self.blocks:
            block_fn = functools.partial(block, seq_mask=mask, pair_mask=pair_mask)
            if self.training and not is_recycling:
                seq_act, pair_act = checkpoint(block_fn, seq_act, pair_act)
            else:
                seq_act, pair_act = block_fn(seq_act, pair_act)

        return seq_act, pair_act

class SeqformerIteration(nn.Module):
    def __init__(self, config, seq_channel, pair_channel):
        super().__init__()
        c = config

        self.outer_product_mean = OuterProductMean(c.outer_product_mean, seq_channel, pair_channel)

        self.triangle_multiplication_outgoing = TriangleMultiplication(c.triangle_multiplication_outgoing, pair_channel)
        self.triangle_multiplication_incoming = TriangleMultiplication(c.triangle_multiplication_incoming, pair_channel)
        self.triangle_attention_starting_node = TriangleAttention(c.triangle_attention_starting_node, pair_channel)
        self.triangle_attention_ending_node = TriangleAttention(c.triangle_attention_ending_node, pair_channel)
        self.pair_transition = Transition(c.pair_transition, pair_channel)

        self.seq_left_transition = Transition(c.seq_transition, pair_channel, seq_channel)
        self.seq_right_transition = Transition(c.seq_transition, pair_channel, seq_channel)

        self.config = config

    def forward(self, seq_act, pair_act, seq_mask, pair_mask=None):
        """
        seq_act: (b l c)
        pair_act: (b l l c)
        seq_mask: (b l)
        """
        c = self.config

        def dropout_fn(input_act, act, config):
            if self.training and config.dropout_rate > 0.:
                if config.shared_dropout:
                    if config.orientation == 'per_row':
                        broadcast_dim = 1
                    else:
                        broadcast_dim = 2
                else:
                    broadcast_dim = None
                act = apply_dropout(act, config.dropout_rate,
                        is_training=True, broadcast_dim=broadcast_dim)
            return input_act + act

        seq_act = dropout_fn(seq_act, torch.sum(self.seq_left_transition(pair_act, pair_mask) * pair_mask[...,None], dim=1), c.seq_transition)
        seq_act = dropout_fn(seq_act, torch.sum(self.seq_right_transition(pair_act, pair_mask) * pair_mask[...,None], dim=2), c.seq_transition)

        pair_act = pair_act + self.outer_product_mean(seq_act, seq_mask)

        pair_act = dropout_fn(
                pair_act, self.triangle_multiplication_outgoing(pair_act, seq_mask, pair_mask=pair_mask), c.triangle_multiplication_outgoing)
        pair_act = dropout_fn(
                pair_act, self.triangle_multiplication_incoming(pair_act, seq_mask, pair_mask=pair_mask), c.triangle_multiplication_incoming)
        pair_act = dropout_fn(
                pair_act, self.triangle_attention_starting_node(pair_act, seq_mask, pair_mask=pair_mask), c.triangle_attention_starting_node)
        pair_act = dropout_fn(
                pair_act, self.triangle_attention_ending_node(pair_act, seq_mask, pair_mask=pair_mask), c.triangle_attention_ending_node)
        pair_act = pair_act + self.pair_transition(pair_act, seq_mask)

        return seq_act, pair_act

# AlphaFold version
class OuterProductMean(nn.Module):
    def __init__(self, config, num_in_channel, num_out_channel):
        super().__init__()

        c = config
        self.norm = LayerNorm(num_in_channel)
        self.left_proj = Linear(num_in_channel, c.num_outer_channel, init='linear')
        self.right_proj = Linear(num_in_channel, c.num_outer_channel, init='linear')

        self.out_proj = Linear(c.num_outer_channel * c.num_outer_channel, num_out_channel, init='final')

    def forward(self, act, mask):
        """
        act: (b l c)
        mask: (b l)
        """
        mask = rearrange(mask, 'b l -> b l ()')
        act = self.norm(act)
        left_act = mask * self.left_proj(act)
        right_act = mask * self.right_proj(act)

        act = torch.einsum('b i c, b j d -> b i j c d', left_act, right_act)
        act = rearrange(act, 'b i j c d -> b i j (c d)')

        act = self.out_proj(act)

        return act

class Transition(nn.Module):
    def __init__(self, config, num_in_channel, num_out_channel=None):
        super().__init__()

        c = config

        if num_out_channel is None:
            num_out_channel = num_in_channel

        intermediate_channel = num_in_channel * c.num_intermediate_factor

        self.norm = LayerNorm(num_in_channel)
        self.proj_in = Linear(num_in_channel, intermediate_channel, init='linear')
        self.proj_out = Linear(intermediate_channel, num_out_channel, init='final')

    def forward(self, act, mask):
        act = self.proj_in(self.norm(act))
        act = F.relu(act)
        act = self.proj_out(act)

        return act
