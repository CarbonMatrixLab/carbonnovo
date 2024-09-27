import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from carbonmatrix.model import atom
from carbonmatrix.model.utils import l2_normalize
from carbonmatrix.model.common_modules import Linear
from carbonmatrix.model.common_modules import get_lora_config

class ResNetBlock(nn.Module):
    def __init__(self, dim, lora_config={}):
        super().__init__()

        self.net = nn.Sequential(
                nn.ReLU(),
                Linear(dim, dim, init='relu', **lora_config),
                nn.ReLU(),
                Linear(dim, dim, init='final', **lora_config))

    def forward(self, act):
        return act + self.net(act)

class TorsionModule(nn.Module):
    def __init__(self, config, num_in_channel, num_in_initial_channel):
        super().__init__()
        c = config
        lora_config = get_lora_config(c)

        self.proj_act = nn.Sequential(
                nn.ReLU(),
                Linear(num_in_channel, c.num_channel, init='linear', **lora_config),
                )
        self.proj_init_act = nn.Sequential(
                nn.ReLU(),
                Linear(num_in_initial_channel, c.num_channel, init='linear', **lora_config),
                )
        self.blocks = nn.Sequential(
                *[ResNetBlock(c.num_channel, lora_config) for _ in range(c.num_residual_block)])

        # (preomega, phi, psi)
        self.projection = Linear(c.num_channel, 7 * 2, init='linear', **lora_config)

    def forward(self, act, init_act):

        act = self.proj_act(act) + self.proj_init_act(init_act)
        act = self.blocks(act)

        angles = rearrange(self.projection(F.relu(act)), '... (n d)->... n d', d=2)

        return angles

class MultiRigidSidechain(nn.Module):
    def __init__(self, config, num_in_seq_channel):
        super().__init__()
        c = config

        self.torsion_module = TorsionModule(c.torsion, c.num_channel, c.num_channel)

        self.config = config

    def forward(self, seq, backb_to_global, representations_list, compute_atom_pos=True):
        assert len(representations_list) == 2

        # Shape: (num_res, 14).
        unnormalized_angles = self.torsion_module(*representations_list)
        angles = l2_normalize(unnormalized_angles, dim=-1)

        outputs = {
                'angles_sin_cos': angles, # (N, 7, 2)
                'unnormalized_angles_sin_cos':  unnormalized_angles,  # (N, 7, 2)
                }

        if not compute_atom_pos:
            return outputs
        # (N, 8)
        all_frames_to_global = atom.torsion_angles_to_frames(seq, backb_to_global, angles)

        # (N, 14)
        pred_positions = atom.frames_and_literature_positions_to_atom14_pos(seq, all_frames_to_global)

        outputs.update({
            'atom_pos': pred_positions,  # r3.Vecs (N, 14)
            'frames': all_frames_to_global,  # r3.Rigids (N, 8)
            })

        return outputs
