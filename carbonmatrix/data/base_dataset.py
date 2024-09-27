import os
import functools
import math
import pathlib
import random

import numpy as np
import torch
from torch.nn import functional as F


from carbonmatrix.common import residue_constants
from carbonmatrix.common.operator import pad_for_batch, pad_for_batch2
from carbonmatrix.data.seq import str_seq_to_index
from carbonmatrix.data.transform_factory import TransformFactory

class Cluster(object):
    def __init__(self, names):
        self.names = names
        self.idx = 0
        assert len(names) > 0

    def get_next(self):
        item = self.names[self.idx]
        self.idx += 1
        if self.idx == len(self.names):
            self.idx = 0
        return item

    def __expr__(self):
        return self.names[self.idx]

    def __str__(self):
        return self.names[self.idx]

def parse_cluster(file_name):
    ret = []
    with open(file_name) as f:
        for line in f:
            items = line.strip().split()
            ret.append(Cluster(names=items))
    return ret

class TransformedDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, feats, device, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)

        self.transform_factory = TransformFactory(feats)
        self.device = device
        
    def set_epoch(self, epoch):
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
    
    def __iter__(self,):
        for batch in super().__iter__():
            batch = {k : v.to(device=self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            yield self.transform_factory(batch)

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, max_seq_len=None):
        super().__init__()

        self.max_seq_len = max_seq_len

    def __get_item(self, idx):
        raise NotImplementedError('_get_next_seq not implemented')

    def _create_seq_data(self, name, str_seq):
        multimer_str_seq = str_seq.split(':')

        str_seq = ''.join(multimer_str_seq)

        N = len(str_seq)
        return dict(
                name = name,
                str_seq = str_seq,
                seq = str_seq_to_index(str_seq),
                mask = np.ones((N,), dtype=np.bool_),
                multimer_str_seq = multimer_str_seq,
                )
    
    def __getitem__(self, idx):
        item = self._get_item(idx)

        ret = self._create_seq_data(item['name'], item['seq'])
        if 'meta' in item:
            ret.update(meta=item['meta'])
        
        for k, v in ret.items():
            ret[k] = torch.from_numpy(v) if isinstance(v, np.ndarray) else v

        return ret

def collate_fn_seq(batch):
    def _gather(n):
        return [b[n] for b in batch]
    
    name = _gather('name')
    str_seq = _gather('str_seq')
    multimer_str_seq = _gather('multimer_str_seq')
    chain_id = torch.zeros((len(str_seq),), dtype=torch.int32)
    meta = {} if 'meta' not in batch[0].keys() else _gather('meta')

    max_len = max(tuple(len(s) for s in str_seq))
        
    return dict(
            name=name,
            str_seq = str_seq,
            multimer_str_seq = multimer_str_seq,
            seq = pad_for_batch(_gather('seq'), max_len, residue_constants.unk_restype_index),
            mask = pad_for_batch(_gather('mask'), max_len, 0),
            batch_len = max_len, 
            meta=meta,
#            chain_id=pad_for_batch2(chain_id, max_len, 'msk'),
            )
