import pickle

import numpy as np
import torch
from utils import convert_to_tensor
import random
import torch.nn.functional as F

def compute_mean_std(states: np.ndarray, eps: float):
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std

def denormalize_states(normalized_states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return normalized_states * std + mean

class Dataset_C(torch.utils.data.Dataset):
    """Dataset class."""
    def __init__(self, path, context_encoder, config, normal=False):
        self.shuffle = config['shuffle']
        self.horizon = config['horizon']
        self.store_gpu = config['store_gpu']
        self.config = config
        self.context_encoder = context_encoder
        self.mean = 0
        self.std = 1
        # if path is not a list
        if not isinstance(path, list):
            path = [path]

        self.trajs = []
        for p in path:
            with open(p, 'rb') as f:
                self.trajs += pickle.load(f)

        context_states = []
        context_next_states = []
        context_actions = []
        context_rewards = []
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        goals = []
        contexts = []

        for traj in self.trajs:
            context_states.append(traj['context_states'])
            context_next_states.append(traj['context_next_states'])
            context_actions.append(traj['context_actions'])
            context_rewards.append(traj['context_rewards'])
            states.append(traj['states'])
            actions.append(traj['actions'])
            next_states.append(traj['next_states'])
            rewards.append(traj['rewards'])
            dones.append(traj['dones'])
            goals.append(traj['goal'])
            state_segment = F.pad(convert_to_tensor(traj['context_states'], store_gpu=self.store_gpu, device=self.config['device']), (0, 0, self.config['context_horizon']-1, 0, 0, 0)).unfold(1, self.config['context_horizon'], 1)[:, :self.horizon, :, :].permute(0, 1, 3, 2)
            action_segment = F.pad(convert_to_tensor(traj['context_actions'], store_gpu=self.store_gpu, device=self.config['device']), (0, 0, self.config['context_horizon']-1, 0, 0, 0)).unfold(1, self.config['context_horizon'], 1)[:, :self.horizon, :, :].permute(0, 1, 3, 2)
            reward_segment = F.pad(convert_to_tensor(traj['context_rewards'][:, :, None], store_gpu=self.store_gpu, device=self.config['device']), (0, 0, self.config['context_horizon']-1, 0, 0, 0)).unfold(1, self.config['context_horizon'], 1)[:, :self.horizon, :, :].permute(0, 1, 3, 2)
            context = self.context_encoder(state_segment.reshape(-1, state_segment.shape[-2], state_segment.shape[-1]), action_segment.reshape(-1, action_segment.shape[-2], action_segment.shape[-1]), reward_segment.reshape(-1, reward_segment.shape[-2], reward_segment.shape[-1])).cpu().numpy()
            contexts.append(context.reshape(state_segment.shape[0], state_segment.shape[1], context.shape[-1]))

        contexts = np.array(contexts)
        states = np.array(states).reshape(-1, states[0].shape[-1])
        actions = np.array(actions).reshape(-1, actions[0].shape[-1])
        next_states = np.array(next_states).reshape(-1, states[0].shape[-1])
        rewards = np.array(rewards).reshape(-1, 1)
        dones = np.array(dones).reshape(-1, 1)
        goals = np.array(goals).reshape(-1, 1)

        if normal:
            self.mean, self.std = compute_mean_std(states, eps=1e-6)
            self.mean = convert_to_tensor(self.mean, store_gpu=self.store_gpu, device=self.config['device'])
            self.std = convert_to_tensor(self.std, store_gpu=self.store_gpu, device=self.config['device'])

        self.dataset = {
            'states': states,
            'actions': actions,
            'next_states': next_states,
            'rewards': rewards,
            'dones': dones,
            'goals': goals,
            "zeros": None
            }

        self.zeros = np.zeros(
            config['state_dim'] ** 2 + config['action_dim'] + 1
        )
        self.promt_trajs = {}
        self.promt_trajs['contexts'] = convert_to_tensor(contexts, store_gpu=self.store_gpu, device=self.config['device'])

    def __len__(self):
        'Denotes the total number of samples'
        return self.dataset['states'].shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        res = {
            'states': self.dataset['states'][index],
            'actions': self.dataset['actions'][index],
            'next_states': self.dataset['next_states'][index],
            'rewards': self.dataset['rewards'][index],
            'dones': self.dataset['dones'][index],
            'goals': self.dataset['goals'][index],
            'zeros': self.zeros,
        }
        return res


class ContextDataset(torch.utils.data.Dataset):
    """Dataset class."""
    def __init__(self, path, config):
        self.shuffle = config['shuffle']
        self.horizon = config['horizon']
        self.context_horizon = config['context_horizon']
        self.store_gpu = config['store_gpu']
        self.config = config

        # if path is not a list
        if not isinstance(path, list):
            path = [path]

        self.trajs = []
        for p in path:
            with open(p, 'rb') as f:
                self.trajs += pickle.load(f)

        context_states = []
        context_actions = []
        context_next_states = []
        context_rewards = []
        context_terminals=[]
        goals = []

        for traj in self.trajs:
            context_states.append(traj['context_states'])
            context_actions.append(traj['context_actions'])
            context_next_states.append(traj['context_next_states'])
            context_rewards.append(traj['context_rewards'])
            terminals_matrix = np.full(traj['context_rewards'].shape, False)
            terminals_matrix[:,-1] = True
            context_terminals.append(terminals_matrix)
            goals.append(traj['goal'])

        context_states = np.array(context_states)
        context_actions = np.array(context_actions)
        context_next_states = np.array(context_next_states)
        context_rewards = np.array(context_rewards)
        context_terminals = np.array(context_terminals)
        goals = np.array(goals)
        if len(context_rewards.shape) < 4:
            context_rewards = context_rewards[:, :, :, None]
        if len(context_terminals.shape) < 4:
            context_terminals = context_terminals[:,:, :, None]

        self._states = context_states.reshape(-1, context_states.shape[-1])
        self._actions = context_actions.reshape(-1, context_actions.shape[-1])
        self._next_states = context_next_states.reshape(-1, context_next_states.shape[-1])
        self._rewards = context_rewards.reshape(-1, context_rewards.shape[-1])
        self._terminals = context_terminals.reshape(-1, context_terminals.shape[-1])
        self._goals = goals.reshape(-1, goals.shape[-1])

        self.states = self._states
        self.actions =self._actions
        self.next_states = self._next_states
        self.rewards = self._rewards
        self.terminals = self._terminals
        self.goals = self._goals

        self.parse_trajectory_segment(horizon=self.context_horizon)

    def __len__(self):
        'Denotes the total number of samples'
        assert self.states.shape[0] == self.states_segment.shape[0]
        return self.states.shape[0]

    def __getitem__(self, index):
        return (
            self.states[index],
            self.actions[index],
            self.rewards[index],
            self.next_states[index],
            self.terminals[index],
            self.goals[index]
        ), (
            self.states_segment[index],
            self.actions_segment[index],
            self.rewards_segment[index],
        ),

    def parse_trajectory_segment(self, horizon):
        states = self._states
        actions = self._actions
        rewards = self._rewards
        terminals = self._terminals

        states_segment, actions_segment, rewards_segment = [], [], []
        initial_state_idx = 0
        for idx in range(states.shape[0]):
            ### the context for the current state
            start_idx = max(0, idx-horizon, initial_state_idx)
            if initial_state_idx == idx:    # the initial state of a trajectory
                state_seg = np.zeros((horizon, states.shape[1]))
                action_seg = np.zeros((horizon, actions.shape[1]))
                reward_seg = np.zeros((horizon, rewards.shape[1]))
            else:
                state_seg = states[start_idx : idx]
                action_seg = actions[start_idx : idx]
                reward_seg = rewards[start_idx : idx]

            length_gap = horizon - state_seg.shape[0]
            states_segment.append(np.pad(state_seg, ((length_gap, 0),(0, 0))))
            actions_segment.append(np.pad(action_seg, ((length_gap, 0),(0, 0))))
            rewards_segment.append(np.pad(reward_seg, ((length_gap, 0),(0, 0))))

            if terminals[idx]:
                initial_state_idx = idx + 1

        self.states_segment = np.stack(states_segment, axis=0)
        self.actions_segment = np.stack(actions_segment, axis=0)
        self.rewards_segment = np.stack(rewards_segment, axis=0)
