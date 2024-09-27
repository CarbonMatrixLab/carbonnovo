from inspect import isfunction

from einops import rearrange
import numpy as np

import torch
from torch.nn import functional as F

from carbonmatrix.data.seq import create_masked_token, esm_alphabet
from carbonmatrix.common import residue_constants
from carbonmatrix.data.transform_factory import registry_transform
from carbonmatrix.model.utils import batched_select
from carbonmatrix.common.operator import pad_for_batch

@registry_transform
def make_masked_token(batch):
    label_seq = batch['esm_seq']
    N, L = label_seq.shape
    device = label_seq.device

    str_seq = batch['str_seq']
    label_mask = np.zeros((N, L), dtype=np.bool_)
    seq = np.full((N, L), esm_alphabet.padding_idx, dtype=np.int32)
    for i, str_seq_ in enumerate(str_seq):
        n = len(str_seq_) + 2
        seq_, _, label_mask_ = create_masked_token(str_seq_)
        seq[i, :n] = seq_
        label_mask[i, :n] = label_mask_

    batch.update(
        esm_seq = torch.tensor(seq, device=device),
        label_seq = label_seq,
        label_seq_mask = torch.tensor(label_mask, device=device)
        )

    return batch

@registry_transform
def make_seq_weight(batch):
    device = batch['seq'].device
    assert ('meta' in batch)

    weight = torch.tensor([x['weight'] for x in batch['meta']], device=device)
    batch.update(weight=weight)

    return batch
