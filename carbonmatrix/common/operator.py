import torch
from torch.nn import functional as F

import logging
import os

import numpy as np

from Bio.PDB.Chain import Chain
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Model import Model as PDBModel
from Bio.PDB.PDBIO import PDBIO

import torch
from torch.nn import functional as F

from carbonmatrix.common import residue_constants


def pad_for_batch(items, pad_len, pad_value):
    aux_shape = 2 * (items[0].dim() - 1) * [0]
    return torch.stack([F.pad(xx, aux_shape + [0, pad_len - xx.shape[0]], value=pad_value) for xx in items], dim=0)

def pad_for_batch2(items, batch_length, dtype):
    """Pad a list of items to batch_length using values dependent on the item type.

    Args:
        items: List of items to pad (i.e. sequences or masks represented as arrays of
            numbers, angles, coordinates, pssms).
        batch_length: The integer maximum length of any of the items in the input. All
            items are padded so that their length matches this number.
        dtype: A string ('seq', 'msk', 'crd') reperesenting the type of
            data included in items.

    Returns:
         A padded list of the input items, all independently converted to Torch tensors.
    """
    batch = []
    if dtype == 'seq':
        for seq in items:
            z = torch.ones(batch_length - seq.shape[0], dtype=seq.dtype) * residue_constants.unk_restype_index
            c = torch.cat((seq, z), dim=0)
            batch.append(c)
    elif dtype == 'msk':
        # Mask sequences (1 if present, 0 if absent) are padded with 0s
        for msk in items:
            z = torch.zeros(batch_length - msk.shape[0], dtype=msk.dtype)
            c = torch.cat((msk, z), dim=0)
            batch.append(c)
    elif dtype == "crd":
        for item in items:
            z = torch.zeros((batch_length - item.shape[0],  item.shape[-2], item.shape[-1]), dtype=item.dtype)
            c = torch.cat((item, z), dim=0)
            batch.append(c)
    elif dtype == "crd_msk":
        for item in items:
            z = torch.zeros((batch_length - item.shape[0],  item.shape[-1]), dtype=item.dtype)
            c = torch.cat((item, z), dim=0)
            batch.append(c)
    elif dtype == "ebd":
        for item in items:
            z = torch.zeros((batch_length - item.shape[0],  item.shape[-1]), dtype=item.dtype,
device = item.device)
            c = torch.cat((item, z), dim=0)
            batch.append(c)
    elif dtype == "pair":
        for item in items:
            c = F.pad(item, (0, 0, 0, batch_length - item.shape[-2], 0, batch_length - item.shape[-2]))
            batch.append(c)
    else:
        raise ValueError('Not implemented yet!')
    batch = torch.stack(batch, dim=0)
    return batch

