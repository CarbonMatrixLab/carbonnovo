import functools

import torch

_transform_factory = {}

def registry_transform(fn):
    global _transform_factory
    _transform_factory[fn.__name__] = fn

    return fn

class TransformFactory:
    def __init__(self, config):
        self.config = config

    def _transform(self, batch):
        for fn, kwargs in self.config.items():

            batch = _transform_factory[fn](batch, **kwargs)
        return batch

    def __call__(self, batch):
        return self._transform(batch)
