import torch

from typing import Tuple

from esm import rotary_embedding as E
from carbonmatrix.model.utils import batched_select

def apply_rotary_pos_emb_with_index(x, cos, sin, index):
    cos = batched_select(cos, index)
    sin = batched_select(sin, index)

    return (x * cos) + (E.rotate_half(x) * sin)

class RotaryEmbedding(E.RotaryEmbedding):
    def __init__(self, dim: int, *_, **__):
        super().__init__(dim)

        self._seq_len_cached = 0

    def _update_cos_sin_tables_with_index(self, index):
        with torch.no_grad():
            seq_len = torch.max(index).item() + 1

        device = index.device

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len > self._seq_len_cached:#or self._cos_cached.device != device:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(device)

            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

        return self._cos_cached, self._sin_cached


    def forward(self, q: torch.Tensor, k: torch.Tensor, index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables_with_index(index)

        return (
            apply_rotary_pos_emb_with_index(q, self._cos_cached, self._sin_cached, index),
            apply_rotary_pos_emb_with_index(k, self._cos_cached, self._sin_cached, index),
        )
