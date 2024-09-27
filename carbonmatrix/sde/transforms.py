import torch
from torch.nn import functional as F

from carbonmatrix.model import r3
from carbonmatrix.data.transform_factory import registry_transform
from carbonmatrix.sde.se3_diffuser import SE3Diffuser
from carbonmatrix.model.quat_affine import matrix_to_quaternion

@registry_transform
def make_t(batch, is_training=True):
    rot, tran = batch['rigidgroups_gt_frames']
    bs, device = rot.shape[0], rot.device

    eps = 1e-3
    t = torch.rand(bs, device=device) * (1.0 - eps) + eps
    # t = torch.full((bs,), 0.1, device=device)
    batch.update(t=t)

    return batch

@registry_transform
def make_gt_score(batch, se3_conf):
    t = batch['t']

    diffuser = SE3Diffuser.get(se3_conf)

    rigids_bb = r3.rigids_op(batch['rigidgroups_gt_frames'], lambda x: x[:,:,0])
    rot, tran = rigids_bb

    quat = matrix_to_quaternion(rot)

    batch.update(diffuser.forward_marginal((quat, tran), t))

    #batch.update({'gt_rot_so3vec' : gt_rot_so3vec})

    return batch

@registry_transform
def make_sample_ref(batch, se3_conf):
    diffuser = SE3Diffuser.get(se3_conf)

    seq = batch['seq']
    samples_shape = list(batch['seq'].shape)

    t = torch.full(samples_shape[:1], 1.0 - 1e-2, device=seq.device)

    ref = diffuser.sample_ref(t, samples_shape)

    batch.update(t=t, rigids_t=ref)

    return batch
