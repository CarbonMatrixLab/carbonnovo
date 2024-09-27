from os.path import basename, splitext
from collections import OrderedDict
import numpy as np
import pdb
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Chain import Chain as PDBChain
from Bio.PDB.Residue import Residue
from Bio.PDB.vectors import Vector as Vector, calc_dihedral

from carbonmatrix.common import residue_constants

def extract_chain_subset(orig_chain, res_ids):
    chain = PDBChain(orig_chain.id)
    for residue in orig_chain:
        if residue.id in res_ids:
            residue.detach_parent()
            chain.add(residue)
    return chain

def parse_pdb(pdb_file, model=0):
    parser = PDBParser()
    structure = parser.get_structure(get_pdb_id(pdb_file), pdb_file)
    return structure[model]

def make_stage1_feature(structure):
    N = len(structure)

    coords = np.zeros((N, 14, 3), dtype=np.float32)
    coord_mask = np.zeros((N, 14), dtype=bool)
    str_seq = []

    for seq_idx, residue in enumerate(structure):
        aa = residue_constants.restype_3to1.get(residue.resname, 'X')
        str_seq.append(aa)
        res_atom14_list = residue_constants.restype_name_to_atom14_names.get(residue.resname,
                residue_constants.restype_name_to_atom14_names['UNK'])

        for atom in residue.get_atoms():
            #if atom.id not in ['CA', 'C', 'N', 'O']:
            #    continue
            atom14idx = res_atom14_list.index(atom.id)
            coords[seq_idx, atom14idx] = atom.get_coord()
            coord_mask[seq_idx, atom14idx]= True

    feature = dict(
            str_seq=''.join(str_seq),
            residx = np.arange(N),
            coords=coords,
            coord_mask=coord_mask)

    return feature

def make_stage1_feature_from_pdb(pdb_file):
    struc = parse_pdb(pdb_file)
    N = len(list(struc.get_chains()))
    assert (N in [1, 2])

    if N == 1:
        return make_stage1_feature(struc['A'])

    A = make_stage1_feature(struc['A'])
    B = make_stage1_feature(struc['B'])

    return dict(
            str_seq= A['str_seq'] + B['str_seq'],
            residx = np.concatenate([A['residx'], B['residx'] + A['residx'].shape[0] + residue_constants.residue_chain_index_offset], axis=0),
            coords=np.concatenate([A['coords'], B['coords']], axis=0),
            coord_mask=np.concatenate([A['coord_mask'], B['coord_mask']], axis=0))

def make_chain_feature(chain):
    N = len(structure)

    coords = np.zeros((N, 14, 3), dtype=np.float32)
    coord_mask = np.zeros((N, 14), dtype=bool)
    str_seq = []

    for seq_idx, residue in enumerate(structure):
        aa = residue_constants.restype_3to1.get(residue.resname, 'X')
        str_seq.append(aa)
        res_atom14_list = residue_constants.restype_name_to_atom14_names.get(residue.resname,
                residue_constants.restype_name_to_atom14_names['UNK'])

        for atom in residue.get_atoms():
            atom14idx = res_atom14_list.index(atom.id)
            coords[seq_idx, atom14idx] = atom.get_coord()
            coord_mask[seq_idx, atom14idx]= True

    feature = dict(
            str_seq=''.join(str_seq),
            coords=coords,
            coord_mask=coord_mask)

    return feature

def make_feature_from_pdb(pdb_file):
    struc = parse_pdb(pdb_file)

    chain_num = len(list(struc.get_chains()))
    assert (chain_num == 1)

    feat = make_chain_feature(struc.get_chains[0])

    return feat

def make_feature_from_npz(npz_file, is_ab_feature=False, shuffle_multimer_seq=False):
    #x = np.load(npz_file)

    if not is_ab_feature:
        length = int(npz_file)
        str_seq = ''
        for i in range(length):
            str_seq = str_seq + 'G'
        coord_mask = np.ones((length,14), dtype=bool)
        coords = np.zeros((length, 14, 3),dtype='float32')

        #str_seq = str(x['seq']) if 'seq' in x else str(x['str_seq'])

        #coords = x['coords']
        #coord_mask = x['coord_mask']
    else:
        assert ('heavy_str_seq' in x)
        str_seq = [str(x['heavy_str_seq'])]
        coords = x['heavy_coords']
        coord_mask = x['heavy_coord_mask']

        if 'light_str_seq' in x:
            coords = [coords,  x['light_coords']]
            coord_mask = [coord_mask,  x['light_coord_mask']]
            str_seq.append(str(x['light_str_seq']))

            if shuffle_multimer_seq and np.random.rand() > 0.5:
                coords = coords[::-1]
                coord_mask = coord_mask[::-1]
                str_seq = str_seq[::-1]

            coords = np.concatenate(coords, axis=0)
            coord_mask = np.concatenate(coord_mask, axis=0)

        str_seq = ':'.join(str_seq)

    return dict(
            str_seq = str_seq,
            coords = coords,
            coord_mask = coord_mask)
