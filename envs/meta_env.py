import itertools

import gym
import numpy as np
import torch

from envs.base_env import BaseEnv


class MetaEnvVec(BaseEnv):
    """
    Vectorized Darkroom environment.
    """

    def __init__(self, envs, state_dim, action_dim):
        self.envs = envs
        self.num_envs = len(envs)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def reset(self):
        return [env.reset() for env in self.envs]

    def step(self, actions):
        next_obs, rews, dones = [], [], []
        for action, env in zip(actions, self.envs):
            next_ob, rew, done, _ = env.step(action)
            next_obs.append(next_ob)
            rews.append(rew)
            dones.append(done)
        return next_obs, rews, dones, {}

    def deploy(self, ctrl):
        ob = self.reset()
        obs = []
        acts = []
        next_obs = []
        rews = []
        done = False

        while not done:
            act = ctrl.act(ob)

            obs.append(ob)
            acts.append(act)

            ob, rew, done, _ = self.step(act)
            done = all(done)

            rews.append(rew)
            next_obs.append(ob)

        obs = np.stack(obs, axis=1)
        acts = np.stack(acts, axis=1)
        next_obs = np.stack(next_obs, axis=1)
        rews = np.stack(rews, axis=1)
        return obs, acts, next_obs, rews
