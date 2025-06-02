import numpy as np
import scipy
import torch

from ctrls.ctrl_bandit import Controller

class MetaOptPolicy(Controller):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def reset(self):
        return

    def act(self, state):
        return self.policy.select_action(state, True)


class MetaTransformerController(Controller):
    def __init__(self, model, batch_size=1, device=None):
        self.model = model
        self.state_dim = model.config['state_dim']
        self.action_dim = model.config['action_dim']
        self.horizon = model.horizon
        self.zeros = torch.zeros(
            batch_size, self.state_dim ** 2 + self.action_dim + 1).float().to(device)
        self.temp = 1.0
        self.batch_size = batch_size
        self.device=device

    def act(self, state):
        self.batch['zeros'] = self.zeros

        states = torch.tensor(np.array(state)).float().to(self.device)
        if self.batch_size == 1:
            states = states[None, :]
        self.batch['states'] = states

        actions = self.model(self.batch).cpu().detach().numpy()
        if self.batch_size == 1:
            actions = actions[0]

        return actions
