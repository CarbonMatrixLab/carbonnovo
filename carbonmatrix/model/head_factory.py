import inspect

import torch
from torch import nn
from torch.nn import functional as F

_head_factory = {}

def registry_head(name):
    def wrap(cls):
        global _head_factory
        _head_factory[name] = cls
        return cls
    return wrap

def _valid_head(head_name):
    return head_name in _head_factory

def _create_head(head_name, head_config):
    return _head_factory[head_name](**head_config)

def _update_head_config(head_name, head_config, seq_channel, pair_channel):
    new_config = dict(config = head_config)
    
    args = inspect.getfullargspec(_head_factory[head_name]).args
    if 'num_in_seq_channel' in args:
        new_config.update(num_in_seq_channel = seq_channel)
    
    if 'num_in_pair_channel' in args:
        new_config.update(num_in_pair_channel = pair_channel)
    
    return new_config

class HeadFactory(object):
    def __init__(self,):
        super().__init__()

    @staticmethod
    def from_config(config, seq_channel, pair_channel, parent):
        heads = []
        for head_name, head_config in config.items():
            if not head_config.enabled:
                continue

            assert (_valid_head(head_name))
            
            head_config = _update_head_config(head_name, head_config, seq_channel, pair_channel)
            head = _create_head(head_name, head_config)
            
            if isinstance(parent, nn.Module):
                parent.add_module(head_name, head)

            heads.append((head_name, head))

        return heads
