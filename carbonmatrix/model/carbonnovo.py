import functools
import logging
import random

import torch
from torch import nn

from carbonmatrix.model.seqformer import EmbeddingAndSeqformer
from carbonmatrix.model.head_factory import HeadFactory
from carbonmatrix.model.common_modules import (
        pseudo_beta_fn_v2,
        dgram_from_positions)

class CarbonNovoIteration(nn.Module):
    def __init__(self, config):
        super().__init__()

        c = config

        self.seqformer_module = EmbeddingAndSeqformer(c.embeddings_and_seqformer)

        self.heads = HeadFactory.from_config(
                c.heads,
                seq_channel=c.embeddings_and_seqformer.seq_channel,
                pair_channel=c.embeddings_and_seqformer.pair_channel,
                parent=self)

        self.config = config

    def forward(self, batch, compute_loss = False):
        c = self.config

        seq_act, pair_act = self.seqformer_module(batch)

        representations = {'pair': pair_act, 'seq': seq_act}

        ret = {}

        ret['representations'] = representations

        ret['heads'] = {}

        for name, module in self.heads:
            if compute_loss or name == 'structure_module' or name == 'sequence':
                value = module(ret['heads'], representations, batch)
                if value is not None:
                    ret['heads'][name] = value

        return ret

class CarbonNovo(nn.Module):
    def __init__(self, config):
        super().__init__()

        #self.esm, _ = load_model_and_alphabet_local(config['esm2_model_file'])
        #self.esm.half()

        self.impl = CarbonNovoIteration(config)

        self.config = config

#    def _compute_language_model(self, tokens, residx):
#        repr_layers = list(range(self.config.embeddings_and_seqformer.esm.num_layers + 1))
#
#        with torch.no_grad():
#            results = self.esm(tokens, repr_layers=repr_layers, residx=residx, need_head_weights=False, return_contacts=False)
#
#        ret = {}
#        esm_embed = torch.stack([results['representations'][k][:,1:-1] for k in repr_layers], dim=-1)
#        esm_logits = results['logits'][:,1:-1]
#
#        return esm_embed, esm_logits

    def forward(self, batch, compute_loss=False):
        c = self.config

        seq = batch['seq']

        batch_size, num_residues, device = *seq.shape[:2], seq.device

     #   esm_embed, _ = self._compute_language_model(batch['esm_seq'], batch['residx'])
     #   batch.update(esm_embed = esm_embed)

        def get_prev(ret):
            prev_pseudo_beta = pseudo_beta_fn_v2(batch['seq'], ret['heads']['structure_module']['final_atom_positions'])
            prev_disto_bins = dgram_from_positions(prev_pseudo_beta, **self.config.embeddings_and_seqformer.prev_pos)

            new_prev = {
                    'prev_pos': prev_disto_bins.detach(),
                    'prev_seq': ret['representations']['seq'].detach(),
                    'prev_pair': ret['representations']['pair'].detach()
                    }
            return new_prev

        # Just to adapt to ESMFOLD
        emb_config = c.embeddings_and_seqformer
        if True:#'rigids_t' in batch:
            quat_t, trans_t = batch['rigids_t']
            prev_pos = dgram_from_positions(trans_t, **self.config.embeddings_and_seqformer.prev_pos)
        else:
            prev_pos = torch.zeros([batch_size, num_residues, num_residues], device=device, dtype=torch.int64)
        #prev_pos = torch.zeros([batch_size, num_residues, num_residues], device=device, dtype=torch.int64)

        prev = {
                'prev_pos': prev_pos,
                'prev_seq': torch.zeros([batch_size, num_residues, emb_config.seq_channel], device=device),
                'prev_pair': torch.zeros([batch_size, num_residues, num_residues, emb_config.pair_channel], device=device)
        }
        batch.update(prev)

        if self.training:
            num_recycle = random.randint(0, c.num_recycle-1)
        else:
            num_recycle = c.num_recycle

        with torch.cuda.amp.autocast(enabled=False, dtype=torch.float16):
            with torch.no_grad():
                batch.update(is_recycling=True)
                for i in range(num_recycle+1):
                    ret = self.impl(batch, compute_loss=False)

                    prev = get_prev(ret)
                    batch.update(prev)

        batch.update(is_recycling=False)
        ret = self.impl(batch, compute_loss=compute_loss)

        return ret
