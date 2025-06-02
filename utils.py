import numpy as np
import torch
from torch.utils.data import DataLoader
import math
import functools


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**32) + worker_id
    torch.manual_seed(worker_seed)
    numpy_seed = int(worker_seed % (2**32 - 1))  # Optional, in case you also use numpy in the DataLoader
    np.random.seed(numpy_seed)


def build_meta_data_filename(env, n_envs, config, mode):
    """
    Builds the filename for the darkroom data.
    Mode is either 0: train, 1: test, 2: eval.
    """
    filename_template = 'datasets/trajs_{}.pkl'
    filename = env
    filename += '_H' + str(config['horizon'])
    if mode == 0:
        filename += '_' + config['rollin_type']
        filename += '_train'
    elif mode == 1:
        filename += '_' + config['rollin_type']
        filename += '_test'
    elif mode == 2:
        filename += '_' + config['rollin_type']
        filename += '_test_eval'
    elif mode == 3:
        filename += '_' + config['rollin_type']
        filename += '_train_eval'

    return filename_template.format(filename)

def build_sicql_model_filename(env, config):
    """
    Builds the filename for the darkroom model.
    """
    filename = env
    filename += '_lr' + str(config['lr'])
    filename += '_do' + str(config['dropout'])
    filename += '_embd' + str(config['n_embd'])
    filename += '_n_layer' + str(config['n_layer'])
    filename += '_m_layer' + str(config['m_layer'])
    filename += '_head' + str(config['n_head'])
    filename += '_H' + str(config['horizon'])
    filename += '_seed' + str(config['seed'])
    filename += '_iql_tau' + str(config['iql_tau'])
    return filename

def build_context_model_filename(env, config):
    """
    Builds the filename for the darkroom model.
    """
    filename = env
    filename += '_lr' + str(config['c_lr'])
    filename += '_do' + str(config['dropout'])
    filename += '_c_layer' + str(config['c_layer'])
    filename += '_embd' + str(config['context_hidden_dim'])
    filename += '_context' + str(config['context_dim'])
    filename += '_c_horizon' + str(config['context_horizon'])
    return filename


def convert_to_tensor(x, store_gpu=True, device=None):
    if store_gpu:
        return torch.tensor(np.asarray(x)).float().to(device)
    else:
        return torch.tensor(np.asarray(x)).float()
