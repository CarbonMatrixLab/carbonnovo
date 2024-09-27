import functools
import numpy as np
import pdb
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from carbonmatrix.common import residue_constants
from carbonmatrix.model.common_modules import LayerNorm, Linear, apply_dropout
from carbonmatrix.model.baseformer import (
        SeqAttentionWithPairBias,
        Transition,
        OuterProductMean,
        TriangleMultiplication,
        TriangleAttention,
        )
from carbonmatrix.data.esm import ESMEmbeddingExtractor

class EmbeddingAndSeqformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config

        self.num_token = residue_constants.restype_num + 3

        if c.lm.enabled:
            self.esm = ESMEmbeddingExtractor.get(c.lm.model_path, repr_layer=c.lm.repr_layer)
            self.lm_norm = LayerNorm(c.seq_channel)
            self.lm_proj = Linear(2560, c.seq_channel, init='linear')

        self.proj_rel_pos = torch.nn.Embedding(c.max_relative_feature * 2 + 2, c.pair_channel)

        if c.recycle_features:
            self.prev_seq_norm = LayerNorm(c.seq_channel)
            self.prev_pair_norm = LayerNorm(c.pair_channel)
    
        if c.recycle_pos:
            self.proj_prev_pos = nn.Embedding(c.prev_pos.num_bins, c.pair_channel)

        self.seqformer = Seqformer(c)

        if c.get('timestep_embedder', None) is not None and c.timestep_embedder.enabled:
            self.timestep_embedder = TimestepEmbedder(c.timestep_embedder.embedding_dim, c.seq_channel, c.timestep_embedder.max_positions)

        self.config = config

    def forward(self, batch):
        c = self.config

        seq, mask= batch['seq'], batch['mask']

        batch_size, num_residue = seq.shape[:2]

#        seq_act = self.proj_aa_type(seq)
        x = self.timestep_embedder(batch['t'])
        seq_act = rearrange(x, 'b c -> b () c')
#        print(seq_act)
        #if 'residx' in batch:
        seq_pos = batch['residx'][:,1:-1] - 1
        #else:
        #    seq_pos = torch.tile(torch.arange(num_residue, device=seq.device), [batch_size, 1])

        offset = rearrange(seq_pos, 'b l -> b () l') - rearrange(seq_pos, 'b l -> b l ()')
        rel_pos = torch.clip(offset + c.max_relative_feature, min=0, max=2*c.max_relative_feature) + 1
        pair_act = self.proj_rel_pos(rel_pos)

        if c.lm.enabled:
            if 'batch_pred_seq' in batch:
                prev_aatype = batch['batch_pred_seq']
                lm_embed = self.esm(prev_aatype, device=pair_act.device)
                time_mask = (batch['t']<0.25)
                lm_embed *= time_mask[:,None,None]
                seq_act = seq_act + self.lm_norm(self.lm_proj(lm_embed))
            else:
                lm_embed = torch.zeros([batch_size, num_residue, c.lm.embed_channel], device=pair_act.device)
            seq_act = seq_act + self.lm_norm(self.lm_proj(lm_embed))
        if c.recycle_features:
            if 'prev_seq' in batch:
                seq_act = seq_act + self.prev_seq_norm(batch['prev_seq'])
            if 'prev_pair' in batch:
                pair_act = pair_act + self.prev_pair_norm(batch['prev_pair'])

        if True:#c.recycle_pos and 'prev_pos' in batch:
            pair_act = pair_act + self.proj_prev_pos(batch['prev_pos'])

        seq_act, pair_act = self.seqformer(seq_act, pair_act, mask=mask, is_recycling=batch['is_recycling'])

        return seq_act, pair_act

class SeqformerIteration(nn.Module):
    def __init__(self, config, seq_channel, pair_channel):
        super().__init__()
        c = config

        self.seq_attn = SeqAttentionWithPairBias(c.seq_attention_with_pair_bias, seq_channel, pair_channel)
        self.seq_transition = Transition(c.seq_transition, seq_channel)
        self.outer_product_mean = OuterProductMean(c.outer_product_mean, seq_channel, pair_channel)

        self.triangle_multiplication_outgoing = TriangleMultiplication(c.triangle_multiplication_outgoing, pair_channel)
        self.triangle_multiplication_incoming = TriangleMultiplication(c.triangle_multiplication_incoming, pair_channel)
        self.triangle_attention_starting_node = TriangleAttention(c.triangle_attention_starting_node, pair_channel)
        self.triangle_attention_ending_node = TriangleAttention(c.triangle_attention_ending_node, pair_channel)
        self.pair_transition = Transition(c.pair_transition, pair_channel)

        self.config = config

    def forward(self, seq_act, pair_act, seq_mask):
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

        seq_act = dropout_fn(
                seq_act, self.seq_attn(seq_act, pair_act, seq_mask), c.seq_attention_with_pair_bias)
        seq_act = seq_act + self.seq_transition(seq_act, seq_mask)

        pair_act = pair_act + self.outer_product_mean(seq_act, seq_mask)

        pair_act = dropout_fn(
                pair_act, self.triangle_multiplication_outgoing(pair_act, seq_mask), c.triangle_multiplication_outgoing)
        pair_act = dropout_fn(
                pair_act, self.triangle_multiplication_incoming(pair_act, seq_mask), c.triangle_multiplication_incoming)
        pair_act = dropout_fn(
                pair_act, self.triangle_attention_starting_node(pair_act, seq_mask), c.triangle_attention_starting_node)
        pair_act = dropout_fn(
                pair_act, self.triangle_attention_ending_node(pair_act, seq_mask), c.triangle_attention_ending_node)
        pair_act = pair_act + self.pair_transition(pair_act, seq_mask)

        return seq_act, pair_act

class Seqformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config

        self.blocks = nn.ModuleList([SeqformerIteration(c.seqformer, c.seq_channel, c.pair_channel) for _ in range(c.seqformer_num_block)])

    def forward(self, seq_act, pair_act, mask, is_recycling=True):
        for it, block in enumerate(self.blocks):
            block_fn = functools.partial(block, seq_mask=mask)
            if self.training and not is_recycling and it > 0:
                seq_act, pair_act = checkpoint(block_fn, seq_act, pair_act, use_reentrant=False)
            else:
                seq_act, pair_act = block_fn(seq_act, pair_act)

        return seq_act, pair_act

class TimestepEmbedder(nn.Module):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    def __init__(self, embedding_dim, output_dim, max_positions=10000):
        super().__init__()
        self.max_positions = max_positions

        half_dim = embedding_dim // 2
        emb = np.log(max_positions) / (half_dim - 1)
        self.emb = torch.exp(torch.arange(half_dim, dtype=torch.float32,) * -emb)

        # no lora config for timestep embeddings
        self.proj_out = Linear(embedding_dim, output_dim, init='linear', bias=False)

    def forward(self, timesteps):
        assert len(timesteps.shape) == 1

        # (batch_size, )
        timesteps = rearrange(timesteps * self.max_positions, 'b -> b ()')
        # (embed_dim,)
        emb = rearrange(self.emb.to(device=timesteps.device), 'c -> () c')

        # (batch_size, embed_dim)
        emb = timesteps * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        # (batch_size, seq_channel)
        emb = self.proj_out(emb)

        return emb
