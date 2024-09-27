from inspect import isfunction

from einops import rearrange
import numpy as np

import torch
from torch.nn import functional as F

from carbonmatrix.data.seq import create_esm_seq, esm_alphabet
from carbonmatrix.common import residue_constants
from carbonmatrix.data.transform_factory import registry_transform
from carbonmatrix.model.utils import batched_select
from carbonmatrix.common.operator import pad_for_batch

@registry_transform
def make_restype_atom_constants(batch):
    device = batch['seq'].device

    batch['atom14_atom_exists'] = batched_select(torch.tensor(residue_constants.restype_atom14_mask, device=device), batch['seq'])
    batch['atom14_atom_is_ambiguous'] = batched_select(torch.tensor(residue_constants.restype_atom14_is_ambiguous, device=device), batch['seq'])

    if 'residx_atom37_to_atom14' not in batch:
        batch['residx_atom37_to_atom14'] = batched_select(torch.tensor(residue_constants.restype_atom37_to_atom14, device=device), batch['seq'])

    if 'atom37_atom_exists' not in batch:
        batch['atom37_atom_exists'] = batched_select(torch.tensor(residue_constants.restype_atom37_mask, device=device), batch['seq'])

    return batch

@registry_transform
def make_esm_seq(batch,):
    device = batch['seq'].device
    bs = batch['seq'].shape[0]

    esm_seq = [torch.from_numpy(create_esm_seq(s)) for s in batch['str_seq']]

    max_len = max([x.shape[0] for x in esm_seq])
    padded_esm_seq = pad_for_batch(esm_seq, max_len, esm_alphabet.padding_idx)

    residx = np.tile(np.arange(max_len), (bs, 1))

    for i, multimer_str_seq in enumerate(batch['multimer_str_seq']):
        is_multimer = (len(multimer_str_seq) > 1)
        if not is_multimer:
            continue

        relative_pos, start_pos = 0, 1 + len(multimer_str_seq[0])

        for str_seq in multimer_str_seq[1:]:
            end_pos = start_pos + len(str_seq)

            relative_pos += residue_constants.residue_chain_index_offset
            residx[i, start_pos:end_pos] += relative_pos

            start_pos = end_pos

        residx[i, end_pos:] += residue_constants.residue_chain_index_offset

    batch.update(
            esm_seq = padded_esm_seq.to(device=device),
            residx = torch.tensor(residx, device=device),
            )

    return batch
