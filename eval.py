import argparse
import os
import pickle

import matplotlib.pyplot as plt
import torch
import common_args
from evals import eval_meta
from net import Transformer_C
from dataset import Dataset_C
from utils import (
    build_meta_data_filename,
    build_sicql_model_filename,
    build_context_model_filename,
)
import numpy as np
from meta.src.envs import HalfCheetahVelEnv
from model import RNNContextEncoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    common_args.add_eval_args(parser)

    args = vars(parser.parse_args())
    print("Args: ", args)
    n_envs = args['n_envs']
    horizon = args['horizon']
    H = args['horizon']
    n_embd = args['n_embd']
    context_epoch = args['context_epoch']
    context_dim = args['context_dim']
    context_hidden_dim = args['context_hidden_dim']
    num_epochs = args['context_train_epochs']
    c_lr = args['context_lr']
    c_layer = args['c_layer']
    context_horizon = args['context_horizon']
    beta = args['beta']
    iql_tau = args['iql_tau']
    n_hists = args['hists']
    n_samples = args['samples']
    dim = args['dim']
    state_dim = dim
    action_dim = dim
    n_head = args['head']
    n_layer = args['n_layer']
    m_layer = args['m_layer']
    lr = args['lr']
    epoch = args['epoch']
    freq = args['freq']
    shuffle = args['shuffle']
    dropout = args['dropout']
    var = args['var']
    cov = args['cov']
    test_cov = args['test_cov']
    envname = args['env']
    n_eval = args['n_eval']
    seed = args['seed']
    lin_d = args['lin_d']
    rollin_type = args['rollin_type']
    device = args['device']
    normal = args["normal"]

    tmp_seed = seed
    if seed == -1:
        tmp_seed = 0

    torch.manual_seed(tmp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tmp_seed)
    np.random.seed(tmp_seed)

    dataset_config = {
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
        'rollin_type': rollin_type,
    }
    model_config = {
        'shuffle': shuffle,
        'lr': lr,
        'c_lr': c_lr,
        'context_dim': context_dim,
        'context_hidden_dim': context_hidden_dim,
        'context_horizon': context_horizon,
        'c_layer': c_layer,
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

    if envname == 'HalfCheetahVel-v0':
        with open(f'meta/datasets/{envname}/task_goals.pkl', 'rb') as fp:
            tasks = pickle.load(fp)
        env_ = HalfCheetahVelEnv(tasks=tasks)
        max_action = float(env_.action_space.high[0])
        state_dim = env_.observation_space.shape[0]
        action_dim = env_.action_space.shape[0]
        filename = build_sicql_model_filename(envname, model_config)
        context_filename = build_context_model_filename(envname, model_config)
    else:
        raise NotImplementedError

    config = {
        'shuffle': shuffle,
        'horizon': horizon,
        'context_dim': context_dim,
        'context_horizon': context_horizon,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': n_layer,
        'm_layer': m_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'dropout': dropout,
        'test': True,
        'device': device,
        'max_action': max_action,
        'store_gpu': True,
    }

    tmp_filename = filename

    context_encoder = RNNContextEncoder(state_dim, action_dim, context_dim, context_hidden_dim, c_layer).to(device)
    model_path = f'models/context/{rollin_type}/{context_filename}_epoch{context_epoch}.pt'
    print("encoder_path:", model_path)
    context_encoder.load_state_dict(torch.load(model_path))
    for name, param in context_encoder.named_parameters():
        param.requires_grad = False

    dataset_config = {
        'horizon': horizon,
        'dim': dim,
        'rollin_type': rollin_type,
    }

    if envname in ['HalfCheetahVel-v0']:
        dataset_config.update({'rollin_type': rollin_type})
        path_train = build_meta_data_filename(
            envname, n_envs, dataset_config, mode=0)
        path_test = build_meta_data_filename(
            envname, n_envs, dataset_config, mode=1)
        eval_test_filepath = build_meta_data_filename(
            envname, n_eval, dataset_config, mode=2)
        eval_train_filepath = build_meta_data_filename(
            envname, n_eval*9, dataset_config, mode=3)
        save_filename = filename
    else:
        raise ValueError(f'Environment {envname} not supported')

    train_dataset = Dataset_C(path_train, context_encoder, config, normal=normal)
    test_dataset = Dataset_C(path_test, context_encoder, config, normal=normal)
    train_trajs = train_dataset.promt_trajs
    test_trajs = test_dataset.promt_trajs
    mean, std = train_dataset.mean, train_dataset.std

    config.update({'mean': mean, 'std': std})
    model = Transformer_C(config).to(device)

    with open(eval_test_filepath, 'rb') as f:
        eval_test_trajs = pickle.load(f)
    with open(eval_train_filepath, 'rb') as f:
        eval_train_trajs = pickle.load(f)

    evals_filename = f"{model_config['rollin_type']}/{beta}/evals_epoch{epoch}"
    if not os.path.exists(f'figs/sicql/context_{context_epoch}/c{context_horizon}/{evals_filename}'):
        os.makedirs(f'figs/sicql/context_{context_epoch}/c{context_horizon}/{evals_filename}', exist_ok=True)
    if not os.path.exists(f'figs/sicql/context_{context_epoch}/c{context_horizon}/{evals_filename}/online'):
        os.makedirs(f'figs/sicql/context_{context_epoch}/c{context_horizon}/{evals_filename}/online', exist_ok=True)

    if envname in [ 'HalfCheetahVel-v0']:
        config = {
            'Heps': 40,
            'horizon': horizon,
            'H': H,
            'n_eval': min(20, n_eval),
            'state_dim': state_dim,
            'action_dim': action_dim,
            'device': device
        }
        model_path = f'models/sicql/context_{context_epoch}/c{context_horizon}/{rollin_type}/{beta}/{tmp_filename}_epoch{epoch}.pt'
        print("model_path:", model_path)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        model.eval()
        test_envs = []
        for i_eval in range(len(eval_test_trajs)):
            print(f"Eval env: {eval_test_trajs[i_eval]['goal']}")
            traj = eval_test_trajs[i_eval]
            if args['env'] == "HalfCheetahVel-v0":
                with open(f'meta/datasets/{args["env"]}/task_goals.pkl', 'rb') as fp:
                    tasks = pickle.load(fp)
                env = HalfCheetahVelEnv(tasks=tasks)
                env.reset_task(traj['goal'])
            test_envs.append(env)

        config['n_eval'] = len(eval_test_trajs)
        test_means, test_sems = eval_meta.online_c(test_envs, model, context_encoder, context_horizon, **config)
        plt.savefig(f'figs/sicql/context_{context_epoch}/c{context_horizon}/{evals_filename}/online/{save_filename}_test_online.png')
        plt.clf()

        data = {
            "test_means": test_means,
            "test_sems": test_sems,
        }
        np.savez(f"figs/sicql/context_{context_epoch}/c{context_horizon}/{evals_filename}/{save_filename}_online_data.npz", **data)

