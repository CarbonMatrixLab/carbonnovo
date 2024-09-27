import numpy as np

from esm.data import Alphabet
from carbonmatrix.common import residue_constants

esm_alphabet = Alphabet.from_architecture(name='ESM-1b')

def str_seq_to_index(str_seq, mapping=residue_constants.restype_order_with_x, map_unknown_to_x=True):
    seq = []
    for aa in str_seq:
      if aa not in mapping and not map_unknown_to_x:
          raise ValueError(f'Invalid character in the sequence: {aa}')
      seq.append(mapping.get(aa, mapping['X']))

    return np.array(seq)

def create_esm_seq(str_seq):
    L = len(str_seq)
    seq = np.zeros((L + 2,), dtype=np.int64)
    seq[0] = esm_alphabet.cls_idx
    seq[-1] = esm_alphabet.eos_idx

    for i, a in enumerate(str_seq):
        seq[i+1] = esm_alphabet.get_idx(a)

    return seq

def create_masked_token(str_seq):
    L = len(str_seq)
    seq = np.zeros((L + 2,), dtype=np.int64)
    label_seq = np.zeros((L + 2,), dtype=np.int64)
    label_mask = np.zeros((L + 2,), dtype=np.float32)

    seq[0] = esm_alphabet.cls_idx
    label_seq[0] = esm_alphabet.cls_idx
    seq[-1] = esm_alphabet.eos_idx
    label_seq[-1] = esm_alphabet.eos_idx

    # 15
    # 80, 10, 10
    ru = np.random.uniform(0., 1., (L,))
    for i, (a, r) in enumerate(zip(str_seq, ru)):
        label_seq[i+1] = esm_alphabet.get_idx(a)
        if r < 0.15:
            label_mask[i+1] = 1.0
            if r < 0.12:
                seq[i+1] = esm_alphabet.mask_idx
            elif r < 0.135:
                r_a = np.random.choice(residue_constants.restypes)
                seq[i+1] = esm_alphabet.get_idx(r_a)
            else:
                seq[i+1] = label_seq[i+1]
        else:
            seq[i+1] = label_seq[i+1]

    return seq, label_seq, label_mask
