
import numpy as np

from carbonmatrix.common import residue_constants

def index_to_str_seq(index, mapping=residue_constants.restypes_with_x, map_unknown_to_x=True):
    str_seq = [mapping[i] for i in index]
    return ''.join(str_seq)
