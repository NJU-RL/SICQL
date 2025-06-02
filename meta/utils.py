import math
import torch
import numpy as np
import pickle
import shutil
import torch
import yaml
import numpy as np

from collections import OrderedDict
from os.path import join
from pathlib import Path
from torch.nn import Module, Linear, init, Embedding, LayerNorm
from typing import Dict, Any, List, Optional


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def download_config(env_type: str, return_config_path: bool = False):
    config_path = './configs/args_' + env_type.replace('-', '_') + '.yaml'
    with open(config_path, 'r') as fp:
        args: Dict[str, Any] = yaml.load(fp, Loader=yaml.FullLoader)
    if return_config_path:
        return args, config_path
    else:
        return args


def set_seed(seed: int, env: Any) -> None:
    '''
    Set seed.

    :param seed: seed.
    :param env: the environment.
    '''
    if seed != 0:
        env.action_space.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)






