import torch
import torch.nn as nn
import transformers
transformers.set_seed(0)
from transformers import GPT2Config, GPT2Model
from transformers.activations import ACT2FN
import torch.nn as nn
from torch import nn, Tensor

class MLP(nn.Module):
    def __init__(self, n_embd, final_dim, num_layers, output_activation_fn=None):
        super(MLP, self).__init__()
        layers = []

        for _ in range(num_layers):
            layers.append(nn.Linear(n_embd, n_embd))
            layers.append(nn.ReLU())

        # Final output layer
        layers.append(nn.Linear(n_embd, final_dim))

        if output_activation_fn is not None:
            layers.append(output_activation_fn())

        self.final= nn.Sequential(*layers)

    def forward(self, x):
        return self.final(x)


class Transformer_C(nn.Module):
    """Transformer class."""
    def __init__(self, config):
        super(Transformer_C, self).__init__()
        self.config = config
        self.test = config['test']
        self.horizon = self.config['horizon']
        self.n_embd = self.config['n_embd']
        self.n_layer = self.config['n_layer']
        self.m_layer = self.config['m_layer']
        self.n_head = self.config['n_head']
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']
        self.dropout = self.config['dropout']
        self.device = self.config['device']
        self.max_action = self.config['max_action']
        self.context_dim = self.config['context_dim']
        self.mean = self.config['mean']
        self.std = self.config['std']

        config = GPT2Config(
            n_positions=1*(1 + self.horizon),
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(config)

        self.embed_transition = nn.Linear(
            self.state_dim*2 + self.action_dim + 1, self.n_embd)
        self.embed_context = nn.Linear(
            self.context_dim, self.n_embd)
        if self.max_action == 0:
            self.pred_actions = MLP(self.n_embd, self.action_dim, self.m_layer)
        else:
            self.pred_actions = MLP(self.n_embd, self.action_dim, self.m_layer, nn.Tanh)
        self.pred_v =  MLP(self.n_embd, 1, self.m_layer)
        self.pred_q1 =  MLP(self.n_embd, 1, self.m_layer)
        self.pred_q2 =  MLP(self.n_embd, 1, self.m_layer)

    def forward(self, x, trajs=None):
        zeros = x['zeros'][:, None, :]
        if self.test:
            query_states = x['states'][:, None, :]
            batch_size = x['states'].shape[0]
            next_states = zeros[:, :, :self.state_dim]
            real_acitons = zeros[:, :, :self.action_dim]
        else:
            query_states = x['states'][:, None, :]
            real_acitons = x['actions'][:, None, :]
            next_states = x['next_states'][:, None, :]
            batch_size = x['states'].shape[0]
            n_envs =  trajs['contexts'].shape[1]
            index = torch.randint(0, n_envs, (batch_size,))
            goals = x['goals'].squeeze(1).long()
            x['contexts'] =  trajs['contexts'][goals][torch.arange(batch_size), index]

        query_states = (query_states - self.mean) / self.std
        next_states = (next_states - self.mean) / self.std

        contexts =self.embed_context(x['contexts'])
        transition = torch.cat([query_states, zeros[:, :, :self.action_dim], zeros[:, :, :self.state_dim], zeros[:, :, :1]], dim=-1)
        transition = self.embed_transition(transition)
        stacked_inputs =  torch.cat([transition, contexts], dim=1)
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        if self.max_action == 0:
            a = self.pred_actions(transformer_outputs['last_hidden_state'])
        else:
            a = self.pred_actions(transformer_outputs['last_hidden_state'])*self.max_action
        v = self.pred_v(transformer_outputs['last_hidden_state'])

        with torch.no_grad():
            transition = torch.cat([next_states, zeros[:, :, :self.action_dim], zeros[:, :, :self.state_dim], zeros[:, :, :1]], dim= -1)
            transition = self.embed_transition(transition)
            stacked_inputs =  torch.cat([transition, contexts], dim=1)
            transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
            next_v = self.pred_v(transformer_outputs['last_hidden_state'])

        transition = torch.cat([query_states, real_acitons, zeros[:, :, :self.state_dim], zeros[:, :, :1]], dim= -1)
        transition = self.embed_transition(transition)
        stacked_inputs =  torch.cat([transition, contexts], dim=1)
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        q1 = self.pred_q1(transformer_outputs['last_hidden_state'])
        q2 = self.pred_q2(transformer_outputs['last_hidden_state'])
        if self.test:
            return a[:, -1, :]
        return a[:, 1:, :], q1[:, 1:, :], q2[:, 1:, :], v[:, 1:, :], next_v[:, 1:, :]


