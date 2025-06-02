import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)  # or 'forkserver'

import argparse
import os
import copy
import torch.nn.functional as F
import pickle
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import common_args
import random
from dataset import Dataset_C
from net import Transformer_C
from utils import (
    build_meta_data_filename,
    build_sicql_model_filename,
    build_context_model_filename,
)

from meta.src.envs import HalfCheetahVelEnv
from model import RNNContextEncoder


EXP_ADV_MAX = 1000.0

def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

def soft_update(target, source, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    common_args.add_train_args(parser)
    common_args.add_eval_args(parser)

    args = vars(parser.parse_args())
    print("Args: ", args)
    n_envs = args['n_envs']
    horizon = args['horizon']
    n_embd = args['n_embd']
    context_dim = args['context_dim']
    context_epoch = args['context_epoch']
    context_hidden_dim = args['context_hidden_dim']
    num_epochs = args['context_train_epochs']
    c_lr = args['context_lr']
    c_layer = args['c_layer']
    context_horizon = args['context_horizon']
    iql_tau = args['iql_tau']
    tau = args['tau']
    beta = args['beta']
    discount = args['discount']
    env = args['env']
    n_hists = args['hists']
    n_samples = args['samples']
    dim = args['dim']
    state_dim = dim
    action_dim = dim
    freq = args['freq']
    n_head = args['head']
    n_layer = args['n_layer']
    m_layer = args['m_layer']
    lr = args['lr']
    shuffle = args['shuffle']
    dropout = args['dropout']
    var = args['var']
    cov = args['cov']
    num_epochs = args['num_epochs']
    seed = args['seed']
    lin_d = args['lin_d']
    rollin_type = args['rollin_type']
    device = args["device"]
    normal = args["normal"]

    tmp_seed = seed
    if seed == -1:
        tmp_seed = 0

    if not os.path.exists(f'figs/sicql/context_{context_epoch}/c{context_horizon}/{rollin_type}/{beta}/loss'):
        os.makedirs(f'figs/sicql/context_{context_epoch}/c{context_horizon}/{rollin_type}/{beta}/loss', exist_ok=True)
    if not os.path.exists(f'models/sicql/context_{context_epoch}/c{context_horizon}/{rollin_type}/{beta}'):
        os.makedirs(f'models/sicql/context_{context_epoch}/c{context_horizon}/{rollin_type}/{beta}', exist_ok=True)

    torch.manual_seed(tmp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(tmp_seed)
        torch.cuda.manual_seed_all(tmp_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(tmp_seed)
    random.seed(tmp_seed)

    dataset_config = {
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        "context_horizon": context_horizon,
        'dim': dim,
        'rollin_type': rollin_type,
    }
    model_config = {
        'shuffle': shuffle,
        'c_lr': c_lr,
        'context_dim': context_dim,
        'context_hidden_dim': context_hidden_dim,
        'c_layer': c_layer,
        'context_horizon': context_horizon,
        'lr': lr,
        'dropout': dropout,
        'n_embd': n_embd,
        'n_layer': n_layer,
        'm_layer': m_layer,
        'n_head': n_head,
        'n_envs': n_envs,
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
        'seed': seed,
        'rollin_type': rollin_type,
        'beta': beta,
        'iql_tau': iql_tau,
    }

    if env.startswith('HalfCheetahVel'):
        with open(f'meta/datasets/{env}/task_goals.pkl', 'rb') as fp:
            tasks = pickle.load(fp)
        env_ = HalfCheetahVelEnv(tasks=tasks)
        max_action = float(env_.action_space.high[0])
        state_dim = env_.observation_space.shape[0]
        action_dim = env_.action_space.shape[0]
        dataset_config.update({'rollin_type': rollin_type})
        path_train = build_meta_data_filename(
            env, n_envs, dataset_config, mode=0)
        path_test = build_meta_data_filename(
            env, n_envs, dataset_config, mode=1)
        filename = build_sicql_model_filename(env, model_config)
        context_filename = build_context_model_filename(env, model_config)
        loss_fn = torch.nn.MSELoss(reduction='none')
    else:
        raise NotImplementedError

    config = {
        'context_dim': context_dim,
        'context_horizon': context_horizon,
        'horizon': horizon,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': n_layer,
        'm_layer': m_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'shuffle': shuffle,
        'dropout': dropout,
        'max_action': max_action,
        'test': False,
        'store_gpu': True,
        'device': device
    }


    context_encoder = RNNContextEncoder(state_dim, action_dim, context_dim, context_hidden_dim, c_layer).to(device)
    model_path = f'models/context/{rollin_type}/{context_filename}_epoch{context_epoch}.pt'
    print("encoder_path:", model_path)
    context_encoder.load_state_dict(torch.load(model_path, map_location=device))
    for name, param in context_encoder.named_parameters():
        param.requires_grad = False

    params = {
        'batch_size': 128,
        'shuffle': True,
    }

    log_filename = f'figs/sicql/context_{context_epoch}/c{context_horizon}/{rollin_type}/{beta}/loss/{filename}_logs.txt'
    with open(log_filename, 'w') as f:
        pass
    def printw(string):
        """
        A drop-in replacement for print that also writes to a log file.
        """
        # Use the standard print function to print to the console
        print(string)
        # Write the same output to the log file
        with open(log_filename, 'a') as f:
            print(string, file=f)


    train_dataset = Dataset_C(path_train, context_encoder, config, normal=normal)
    test_dataset = Dataset_C(path_test, context_encoder, config, normal=normal)
    train_trajs = train_dataset.promt_trajs
    test_trajs = test_dataset.promt_trajs
    mean, std = train_dataset.mean, train_dataset.std

    config.update({'mean': mean, 'std': std})
    model = Transformer_C(config).to(device)
    target_model = copy.deepcopy(model).requires_grad_(False).to(device)

    train_loader = torch.utils.data.DataLoader(train_dataset, **params)
    test_loader = torch.utils.data.DataLoader(test_dataset, **params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    printw("Num train batches: " + str(len(train_loader)))
    printw("Num test batches: " + str(len(test_loader)))
    test_iterator = infinite_loader(test_loader)
    train_iterator = infinite_loader(train_loader)

    results_dir = os.path.join(f'runs/{env}/sicql/context_{context_epoch}/c{context_horizon}/{rollin_type}/{beta}/{filename}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    writer = SummaryWriter(results_dir)

    for epoch in tqdm(range(num_epochs+1)):
        # TRAINING
        batch = next(train_iterator)
        batch = {k: torch.tensor(v).float().to(device) for k, v in batch.items()}
        pred_actions, q1, q2, v, next_v = model(batch, train_trajs)
        with torch.no_grad():
            _, target_q1, target_q2, _, _ = target_model(batch, train_trajs)
            target_q =  torch.min(target_q1, target_q2)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, iql_tau)

        rewards = batch['rewards'].unsqueeze(1).expand(-1, pred_actions.shape[1], -1)
        dones = batch['dones'].unsqueeze(1).expand(-1, pred_actions.shape[1], -1)
        targets = rewards + (1.0 - dones.float()) * discount * next_v.detach()

        q_loss = (F.mse_loss(q1, targets) + F.mse_loss(q2, targets)) / 2
        exp_adv = torch.exp(beta*(adv.detach())).clamp(max=EXP_ADV_MAX)
        if env.startswith('darkroom') or env.startswith('miniworld'):
            bc_losses = -torch.sum(batch['actions'].unsqueeze(1).expand(-1, pred_actions.shape[1], -1)* F.log_softmax(pred_actions, dim=-1), dim=-1, keepdim=True)
        else:
            bc_losses = torch.mean((pred_actions - batch['actions'].unsqueeze(1).expand(-1, pred_actions.shape[1], -1)) ** 2, dim=-1, keepdim=True)

        p_loss = torch.mean(exp_adv * bc_losses)
        loss = v_loss + q_loss + p_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        # scheduler.step()
        soft_update(target_model, model, tau)
        writer.add_scalar('q_loss/train', q_loss.item(), epoch)
        writer.add_scalar('actor_loss/train', p_loss.item(), epoch)
        writer.add_scalar('bc_loss/train', bc_losses.mean().item(), epoch)
        writer.add_scalar('value_loss/train', v_loss.item(), epoch)
        # writer.add_scalar("lr", scheduler.get_last_lr()[0],epoch)

        # LOGGING
        if (epoch) % freq == 0:
            torch.save(model.state_dict(),
                       f'models/sicql/context_{context_epoch}/c{context_horizon}/{rollin_type}/{beta}/{filename}_epoch{epoch}.pt')

    torch.save(model.state_dict(), f'models/sicql/context_{context_epoch}/c{context_horizon}/{rollin_type}/{beta}/{filename}.pt')
    print("Done.")