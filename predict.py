import os
import logging
import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from einops import rearrange

from carbonmatrix.model.carbonnovo import CarbonNovo
from carbonmatrix.data.dataset import SeqDataset
from carbonmatrix.data.base_dataset import TransformedDataLoader as DataLoader
from carbonmatrix.sde.se3_diffuser import SE3Diffuser
from carbonmatrix.model import quat_affine
from carbonmatrix.data.base_dataset import collate_fn_seq
from carbonmatrix.common.utils import index_to_str_seq
from carbonmatrix.data.pdbio import save_pdb

def setup(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    log_file = os.path.abspath(os.path.join(cfg.output_dir, 'predict.log'))

    level = logging.DEBUG if cfg.verbose else logging.INFO
    fmt = '%(asctime)-15s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s'

    def _handler_apply(h):
        h.setLevel(level)
        h.setFormatter(logging.Formatter(fmt))
        return h

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file)]

    handlers = list(map(_handler_apply, handlers))

    logging.basicConfig(
        format=fmt,
        level=level,
        handlers=handlers)

    logging.info('-----------------')
    logging.info(f'Arguments: {cfg}')
    logging.info('-----------------')

def to_numpy(x):
    return x.detach().cpu().numpy()

def save_batch_pdb(values, batch, pdb_dir, data_type='general', step=None):
    N = len(batch['str_seq'])
    pred_atom14_coords = to_numpy(values['heads']['structure_module']['final_atom14_positions'])
    names = batch['name']
    str_seqs = batch['str_seq']
    multimer_str_seqs = batch['multimer_str_seq']
    plddt = 0 if 'plddt' not in values else to_numpy(values['plddt'])

    for i in range(N):
        if step is None:
            pdb_file = os.path.join(pdb_dir, f'{names[i]}.pdb')
        else:
            pdb_file = os.path.join(pdb_dir, f'{names[i]}.step{step}.pdb')
        multimer_str_seq = multimer_str_seqs[i]
        str_seq = str_seqs[i]
        chain_ids = ['H', 'L'] if data_type == 'ig' else None
        single_plddt = None #if plddt is None else plddt[i]
        save_pdb(multimer_str_seq, pred_atom14_coords[i, :len(str_seq)], pdb_file, chain_ids)#, single_plddt)

    return

def infer_mrf(init_label, site_repr, pair_repr, site_mask, pair_mask):
    N, C = site_repr.shape
    pair_repr = np.reshape(pair_repr, [N, N, C, C]) * pair_mask[..., None, None]
    deg = np.sum(pair_mask, axis=-1)

    prev_label = np.array(init_label)
    pos = np.argsort(deg)
    for cycle in range(5):
        #shuffle(pos)
        updated_count = 0
        for i in pos:
            if not site_mask[i]:
                continue
            obj = -1000000
            obj_c = -1
            t_lis = np.zeros(C-1)
            for c in range(C - 1):
                t = site_repr[i, c]
                for k in range(N):
                    if pair_mask[i, k]:
                        t += pair_repr[i, k, c, prev_label[k]]
                        t_lis[c] = t
            if True != 0:#t > obj:
                probs = temp_softmax(t_lis, T=0.3)
                obj_c = np.argmax(np.random.multinomial(1, probs))

            if prev_label[i] != obj_c:
                prev_label[i] = obj_c
                updated_count += 1
        if updated_count == 0:
            break
    return prev_label

def temp_softmax(z, T):

    exp_z = np.exp(z/T)
    sum_exp_z = np.sum(exp_z)
    return exp_z / sum_exp_z

def BFevaluate_mrf_one(site_repr, pair_repr, site_mask, pair_mask, cfg, name):
    site_prob = softmax(site_repr)
    label = np.argmax(site_prob, axis=-1)

    pred_str_seq1 = index_to_str_seq(label)

    print(f'>site\n{pred_str_seq1}')
    valid_len = np.sum(site_mask)

    for i in range(8):
        label = infer_mrf(label, site_repr, pair_repr, site_mask, pair_mask)
        pred_str_seq2 = index_to_str_seq(label)
        label[~site_mask] = 20
        print(f'>mrf_{i}\n{pred_str_seq2}')
        with open(os.path.join(cfg.output_dir, name[0] + '.fasta'), 'a') as fw:
            fw.write(f'>{name[0]}@{i+1}\n{pred_str_seq2}\n')

def evaluate_mrf_one( site_repr, pair_repr, site_mask, pair_mask):
    logits = site_repr
    temperature = 0.1
    logits = logits / temperature
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    sampled_indices = np.array([np.random.choice(len(logits[0]), p=probs) for probs in probabilities])
    pred_str_seq1 = index_to_str_seq(sampled_indices)
    print('names=',pred_str_seq1)

    site_prob = softmax(site_repr)
    label = np.argmax(site_prob, axis=-1)
    pred_str_seq1 = index_to_str_seq(label)
    print('names=',pred_str_seq1)

def softmax(x):
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x / np.sum(x, axis=-1, keepdims=True)

def one_hot(a, num_classes=21):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def predict_batch(model, batch, cfg, name):
    data_type = cfg.get('data_type', 'general')
    assert (data_type == 'general')

    bs = batch['seq'].shape[0]
    device = batch['seq'].device

    diffuser = SE3Diffuser.get(cfg.transforms.make_sample_ref.se3_conf)

    def _update_batch(batch, values, t, end_flag):
        mask = batch['mask']
        if end_flag:
            mask = batch['mask']
            pair_mask = batch['pair_distall_mask']
            seq = values['heads']['sequence']['seq_logits'][0].detach().cpu().numpy()
            pair = values['heads']['sequence']['pair_logits'][0].detach().cpu().numpy()
            mask_ = mask[0].detach().cpu().numpy()
            pair_mask = pair_mask[0].detach().cpu().numpy()
            BFevaluate_mrf_one(seq, pair, mask_, pair_mask, cfg, name)
        batch_t = torch.full((bs,), t, device=device)

        pred_rigids_0 = values['heads']['structure_module']['traj'][-1]
        (pred_rot_0, pred_trans_0) = pred_rigids_0
        pred_quat_0 = quat_affine.matrix_to_quaternion(pred_rot_0)

        center = torch.sum(pred_trans_0 * mask[...,None], dim=1) / (torch.sum(mask, dim=1, keepdims=True) + 1e-12)

        pred_trans_0 = pred_trans_0  - rearrange(center, 'b c -> b () c')


        # update
        ret = diffuser.forward_marginal((pred_quat_0, pred_trans_0), batch_t)
        # rigidts_{t_1} which is denoised rigidts_t
        batch.update(t=batch_t, rigids_t=ret['rigids_t'], prev_pos=pred_trans_0)

        return batch

    logging.info('step= 0, time=1.0')
    with torch.no_grad():
        ret = model(batch, compute_loss=True)

    #save_batch_pdb(ret, batch, cfg.output_dir, data_type, step=0)

    dt = 1. / cfg.T
    timesteps = np.linspace(1., 0., cfg.T + 1)[1:-1]
    end_step_time = timesteps[-1]
    end_flag = False
    for i, t in enumerate(timesteps):
        logging.info(f'step= {i+1}, time={t}')

        if t <= end_step_time:
            end_flag = True

        batch = _update_batch(batch, ret, t, end_flag)

        with torch.no_grad():
            ret = model(batch, compute_loss=True)

    # save last step
    save_batch_pdb(ret, batch, cfg.output_dir, data_type)

class GenerativeDataset(SeqDataset):
    def __init__(self, sample_length, sample_number):
        super().__init__()

        self.sample_length = sample_length
        self.sample_number = sample_number

    def __len__(self,):
        return self.sample_number

    def _get_item(self, idx):
        name = f'carbonnovo_{self.sample_length}_{idx}'
        seq = 'G' * self.sample_length

        return dict(name = name, seq = seq)

def predict(cfg):
    dataset = GenerativeDataset(cfg.sample_length, cfg.sample_number)
    collate_fn = collate_fn_seq

    device = cfg.gpu
    torch.cuda.set_device(device)

    test_loader = DataLoader(
            dataset=dataset,
            feats=cfg.transforms,
            device = device,
            collate_fn=collate_fn,
            batch_size=cfg.batch_size,
            drop_last=False,
            )

    ckpt = torch.load(cfg.restore_model_ckpt, map_location='cpu')
    model = CarbonNovo(config = cfg.model)

    model.impl.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    model.to(device)

    for batch in test_loader:
        logging.info('names= >{}'.format(','.join(batch['name'])))
        logging.info('str_len= {}'.format(','.join([str(len(x)) for x in batch['str_seq']])))
        name = batch['name']
        predict_batch(model, batch, cfg, name)


@hydra.main(version_base=None, config_path="./config", config_name="inference_carbonnovo")
def main(cfg : DictConfig):
    setup(cfg)
    predict(cfg)

if __name__ == '__main__':
    main()
