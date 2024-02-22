#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：sac.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fnmatch import fnmatch
from torch.distributions import Normal

from safebench.util.torch_util import CUDA, CPU, kaiming_init
from safebench.scenario.scenario_policy.base_policy import BasePolicy


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        self.fc_std = nn.Linear(256, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.min_val = 1e-3
        # self.apply(kaiming_init)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.tanh(self.fc_mu(x))
        std = self.softplus(self.fc_std(x)) + self.min_val
        return mu, std


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.apply(kaiming_init)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Q(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.fc1 = nn.Linear(state_dim+action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.apply(kaiming_init)

    def forward(self, x, a):
        x = x.reshape(-1, self.state_dim)
        a = a.reshape(-1, self.action_dim)
        x = torch.cat((x, a), -1) # combination x and a
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SAC(BasePolicy):
    name = 'SAC'
    type = 'offpolicy'

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.logger = logger
        self.policy_type = config['scenario_type']

        self.buffer_start_training = config['buffer_start_training']
        self.lr = config['lr']
        self.continue_episode = 0
        self.state_dim = config['scenario_state_dim']
        self.action_dim = config['scenario_action_dim']
        self.min_Val = torch.tensor(config['min_Val']).float()
        self.batch_size = config['batch_size']
        self.update_iteration = config['update_iteration']
        self.gamma = config['gamma']
        self.tau = config['tau']

        self.model_type = config['model_type']
        self.CBV_selection = config['CBV_selection']
        self.model_path = os.path.join(config['ROOT_DIR'], config['model_path'], self.CBV_selection)
        self.scenario_id = config['scenario_id']
        self.agent_info = 'Ego_' + config['agent_policy']

        # create models
        self.policy_net = CUDA(Actor(self.state_dim, self.action_dim))
        self.value_net = CUDA(Critic(self.state_dim))
        self.Q_net = CUDA(Q(self.state_dim, self.action_dim))
        self.Target_value_net = CUDA(Critic(self.state_dim))

        # create optimizer
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.Q_optimizer = optim.Adam(self.Q_net.parameters(), lr=self.lr)

        # define loss function
        self.value_criterion = nn.MSELoss()
        self.Q_criterion = nn.MSELoss()

        # copy parameters
        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.mode = 'train'

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.policy_net.train()
            self.value_net.train()
            self.Q_net.train()
        elif mode == 'eval':
            self.policy_net.eval()
            self.value_net.eval()
            self.Q_net.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')

    @staticmethod
    def info_process(infos):
        CBVs_obs = []
        CBVs_id = []
        env_index = []
        for i, info in enumerate(infos):
            for CBV_id, CBV_obs in info['CBVs_obs'].items():
                CBVs_obs.append(CBV_obs)
                CBVs_id.append(CBV_id)
                env_index.append(i)
        if CBVs_obs:
            info_batch = np.stack(CBVs_obs, axis=0)
            info_batch = info_batch.reshape(info_batch.shape[0], -1)
        else:
            info_batch = None

        return info_batch, CBVs_id, env_index

    def get_init_action(self, state, deterministic=False):
        num_scenario = len(state)
        additional_in = {}
        return [None] * num_scenario, additional_in

    def get_action(self, state, infos, deterministic=False):
        """
            the state came from transition info, so need to be further processed
        """
        state, CBVs_id, env_index = self.info_process(infos)
        scenario_action = [{} for _ in range(len(infos))]
        if state is not None:
            state = CUDA(torch.FloatTensor(state))
            mu, log_sigma = self.policy_net(state)

            if deterministic:
                action = mu
            else:
                sigma = torch.exp(log_sigma)
                dist = Normal(mu, sigma)
                z = dist.sample()
                action = torch.tanh(z)
            for i, (CBV_id, env_id) in enumerate(zip(CBVs_id, env_index)):
                scenario_action[env_id][CBV_id] = CPU(action[i])
        return scenario_action

    def get_action_log_prob(self, state):
        """
            the state came are the scenario obs from the buffer
        """
        batch_mu, batch_log_sigma = self.policy_net(state)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)
        z = dist.sample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + self.min_Val)
        # when action has more than 1 dimensions, we should sum up the log likelihood
        log_prob = torch.sum(log_prob, dim=1, keepdim=True) 
        return action, log_prob, z, batch_mu, batch_log_sigma

    def train(self, buffer, writer, e_i):
        if buffer.buffer_len < self.buffer_start_training:
            return

        for _ in range(self.update_iteration):
            # sample replay buffer
            batch = buffer.sample(self.batch_size)
            bn_s = CUDA(torch.FloatTensor(batch['CBVs_obs'])).reshape(self.batch_size, -1)
            bn_s_ = CUDA(torch.FloatTensor(batch['next_CBVs_obs'])).reshape(self.batch_size, -1)
            bn_a = CUDA(torch.FloatTensor(batch['action']))
            # The reward of the scenario agent
            bn_r = CUDA(torch.FloatTensor(batch['reward'])).unsqueeze(-1)  # [B, 1]
            bn_d = CUDA(torch.FloatTensor(1-batch['done'])).unsqueeze(-1)  # [B, 1]

            target_value = self.Target_value_net(bn_s_)
            next_q_value = bn_r + bn_d * self.gamma * target_value

            expected_value = self.value_net(bn_s)
            expected_Q = self.Q_net(bn_s, bn_a)

            sample_action, log_prob, z, batch_mu, batch_log_sigma = self.get_action_log_prob(bn_s)
            expected_new_Q = self.Q_net(bn_s, sample_action)
            next_value = expected_new_Q - log_prob

            # !!! Note that the actions are sampled according to the current policy, instead of replay buffer. (From original paper)
            V_loss = self.value_criterion(expected_value, next_value.detach())  # J_V
            V_loss = V_loss.mean()

            # Single Q_net this is different from original paper!!!
            Q_loss = self.Q_criterion(expected_Q, next_q_value.detach()) # J_Q
            Q_loss = Q_loss.mean()
            writer.add_scalar("Q loss", Q_loss, e_i)

            log_policy_target = expected_new_Q - expected_value
            pi_loss = log_prob * (log_prob - log_policy_target).detach()
            pi_loss = pi_loss.mean()
            writer.add_scalar("policy loss", pi_loss, e_i)

            # mini batch gradient descent
            self.value_optimizer.zero_grad()
            V_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

            self.Q_optimizer.zero_grad()
            Q_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.Q_net.parameters(), 0.5)
            self.Q_optimizer.step()

            self.policy_optimizer.zero_grad()
            pi_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()

            # soft update
            for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)

    def save_model(self, episode, map_name, buffer):
        states = {
            'policy_net': self.policy_net.state_dict(), 
            'value_net': self.value_net.state_dict(), 
            'Q_net': self.Q_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'Q_optimizer': self.Q_optimizer.state_dict()
        }
        scenario_name = "all" if self.scenario_id is None else 'Scenario' + str(self.scenario_id)
        save_dir = os.path.join(self.model_path, self.agent_info, scenario_name+"_"+map_name)
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f'model.sac.{self.model_type}.{episode:04}.torch')
        self.logger.log(f'>> Saving scenario policy {self.name} model to {os.path.basename(filepath)}', 'yellow')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        # save the replay buffer for the save
        buffer.save_buffer(dir_path=save_dir, filename=f'buffer.{episode:04}.pkl')

    # the loading method corresponds to the episode saving method
    def load_model(self, map_name, episode=None):
        scenario_name = "all" if self.scenario_id is None else 'Scenario' + str(self.scenario_id)
        load_dir = os.path.join(self.model_path, self.agent_info, scenario_name+"_"+map_name)
        if episode is None:
            episode = 0
            for _, _, files in os.walk(load_dir):
                for name in files:
                    if fnmatch(name, "*torch"):
                        cur_episode = int(name.split(".")[-2])
                        if cur_episode > episode:
                            episode = cur_episode

        filepath = os.path.join(load_dir, f'model.sac.{self.model_type}.{episode:04}.torch')
        if os.path.isfile(filepath):
            self.logger.log(f'>> Loading scenario policy {self.name} model from {os.path.basename(filepath)}', 'yellow')
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.value_net.load_state_dict(checkpoint['value_net'])
            self.Target_value_net.load_state_dict(checkpoint['value_net'])
            self.Q_net.load_state_dict(checkpoint['Q_net'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
            self.Q_optimizer.load_state_dict(checkpoint['Q_optimizer'])
            self.continue_episode = episode
        else:
            self.logger.log(f'>> No {self.name} model found at {filepath}', 'red')
            self.continue_episode = 0
