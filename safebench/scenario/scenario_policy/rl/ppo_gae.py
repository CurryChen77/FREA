#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：ppo.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""

import os

import numpy as np
import torch
import torch.nn as nn
from fnmatch import fnmatch
from torch.distributions import Normal
import torch.nn.functional as F
from torch.distributions import Categorical

from safebench.util.torch_util import CUDA, CPU, hidden_init
from safebench.scenario.scenario_policy.base_policy import BasePolicy


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        hidden_dim = 64
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.min_val = 1e-8
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc_mu.weight.data.uniform_(*hidden_init(self.fc_mu))
        self.fc_std.weight.data.uniform_(*hidden_init(self.fc_std))

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.tanh(self.fc_mu(x))
        std = self.softplus(self.fc_std(x)) + self.min_val
        return mu, std

    def select_action(self, state, deterministic):
        with torch.no_grad():
            mu, std = self.forward(state)
            if deterministic:
                action = mu
            else:
                n = Normal(mu, std)
                action = n.sample()
        return CPU(action)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        hidden_dim = 64
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PPO_GAE(BasePolicy):
    name = 'PPO_GAE'
    type = 'onpolicy'

    def __init__(self, config, logger):
        super(PPO_GAE, self).__init__(config, logger)

        self.continue_episode = 0
        self.logger = logger
        self.gamma = config['gamma']
        self.policy_lr = config['policy_lr']
        self.value_lr = config['value_lr']
        self.train_iteration = config['train_iteration']
        self.train_interval = config['train_interval']
        self.state_dim = config['scenario_state_dim']
        self.action_dim = config['scenario_action_dim']
        self.clip_epsilon = config['clip_epsilon']
        self.batch_size = config['batch_size']
        entropy_coeff = config['entropy_coeff']
        self.entropy_coeff = CUDA(torch.tensor(entropy_coeff, dtype=torch.float32))
        self.tau = config['tau']

        self.model_type = config['model_type']
        self.CBV_selection = config['CBV_selection']
        self.model_path = os.path.join(config['ROOT_DIR'], config['model_path'], self.CBV_selection)
        self.scenario_id = config['scenario_id']
        self.agent_info = 'EgoPolicy_' + config['agent_policy'] + "-" + config['agent_obs_type']
        self.safety_network = config['safety_network']

        self.policy = CUDA(PolicyNetwork(state_dim=self.state_dim, action_dim=self.action_dim))
        self.old_policy = CUDA(PolicyNetwork(state_dim=self.state_dim, action_dim=self.action_dim))
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.value = CUDA(ValueNetwork(state_dim=self.state_dim))
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.value_lr)
        self.value_criterion = nn.MSELoss()

        self.mode = 'train'

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.policy.train()
            self.old_policy.train()
            self.value.train()
        elif mode == 'eval':
            self.policy.eval()
            self.old_policy.eval()
            self.value.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')

    def info_process(self, infos):
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
        state, CBVs_id, env_index = self.info_process(infos)
        scenario_action = [{} for _ in range(len(infos))]
        if state is not None:
            state_tensor = CUDA(torch.FloatTensor(state))
            action = self.policy.select_action(state_tensor, deterministic)

            for i, (CBV_id, env_id) in enumerate(zip(CBVs_id, env_index)):
                scenario_action[env_id][CBV_id] = action[i]
        return scenario_action

    def calculate_gae(self, rewards, values, next_values, dones):
        # Calculate deltas
        deltas = [reward + self.gamma * (1 - done) * value_next - value for reward, value, value_next, done in zip(rewards, values, next_values, dones)]

        # Calculate advantages using GAE formula
        advantages = []
        adv = 0
        for delta, done in zip(reversed(deltas), reversed(dones)):
            adv = delta + self.gamma * (1 - done) * self.tau * adv
            advantages.append(adv)
        advantages.reverse()

        # Normalize advantages
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def train(self, replay_buffer, writer, e_i):
        self.old_policy.load_state_dict(self.policy.state_dict())

        # start to train, use gradient descent without batch size
        for K in range(self.train_iteration):
            # TODO for PPO the data should came from the rollout buffer
            batch = replay_buffer.sample(self.batch_size)
            states = CUDA(torch.FloatTensor(batch['CBVs_obs'])).reshape(self.batch_size, -1)
            next_states = CUDA(torch.FloatTensor(batch['next_CBVs_obs'])).reshape(self.batch_size, -1)
            actions = CUDA(torch.FloatTensor(batch['action']))
            rewards = CUDA(torch.FloatTensor(batch['reward'])).unsqueeze(-1)  # [B, 1]
            dones = CUDA(torch.FloatTensor(batch['done'])).unsqueeze(-1)  # [B, 1]

            # calculate advantage using GAE
            with torch.no_grad():
                old_mu, old_std = self.old_policy(states)
                old_n = Normal(old_mu, old_std)
                values = self.value(states)
                next_values = self.value(next_states)
                advantage = self.calculate_gae(rewards, values, next_values, dones)
                reward_sum = values + advantage

            # update policy
            mu, std = self.policy(states)
            n = Normal(mu, std)
            log_prob = n.log_prob(actions)
            old_log_prob = old_n.log_prob(actions)
            ratio = torch.exp(log_prob - old_log_prob)
            L1 = ratio * advantage
            L2 = torch.clamp(ratio, 1.0-self.clip_epsilon, 1.0+self.clip_epsilon) * advantage
            surrogate = torch.min(L1, L2).mean()

            # Entropy bonus
            entropy_bonus = Categorical(log_prob).entropy().mean()

            loss = surrogate + entropy_bonus * self.entropy_coeff

            writer.add_scalar("policy loss", loss, e_i)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # update value function
            value_loss = self.value_criterion(values, reward_sum)
            writer.add_scalar("value loss", value_loss, e_i)
            writer.add_scalars("value net mean value", {"value target": torch.mean(reward_sum), "value net output": torch.mean(values)}, e_i)
            writer.add_scalars("value min-max", {"target-min": torch.min(reward_sum), "net-min": torch.min(values),
                                                "target-max": torch.max(reward_sum), "net-max": torch.max(values)}, e_i)
            self.value_optim.zero_grad()
            value_loss.backward()
            self.value_optim.step()

        # reset buffer
        replay_buffer.reset_buffer()

    def save_model(self, episode, map_name, replay_buffer):
        states = {
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'optim': self.optim.state_dict(),
            'value_optim': self.value_optim.state_dict()
        }
        scenario_name = "all" if self.scenario_id is None else 'scenario' + str(self.scenario_id)
        save_dir = os.path.join(self.model_path, self.agent_info, self.safety_network, scenario_name+"_"+map_name)
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f'model.ppo.{self.model_type}.{episode:04}.torch')
        self.logger.log(f'>> Saving scenario policy {self.name} model to {os.path.basename(filepath)}', 'yellow')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)

    def load_model(self, map_name, episode=None):
        scenario_name = "all" if self.scenario_id is None else 'scenario' + str(self.scenario_id)
        load_dir = os.path.join(self.model_path, self.agent_info, self.safety_network, scenario_name+"_"+map_name)
        if episode is None:
            episode = -1
            for _, _, files in os.walk(load_dir):
                for name in files:
                    if fnmatch(name, "*torch"):
                        cur_episode = int(name.split(".")[-2])
                        if cur_episode > episode:
                            episode = cur_episode
        filepath = os.path.join(load_dir, f'model.ppo.{self.model_type}.{episode:04}.torch')
        if os.path.isfile(filepath):
            self.logger.log(f'>> Loading scenario policy {self.name} model from {os.path.basename(filepath)}', 'yellow')
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.policy.load_state_dict(checkpoint['policy'])
            self.value.load_state_dict(checkpoint['value'])
            self.optim.load_state_dict(checkpoint['optim'])
            self.value_optim.load_state_dict(checkpoint['value_optim'])
            self.continue_episode = episode
        else:
            self.logger.log(f'>> No scenario policy {self.name} model found at {filepath}', 'red')
            self.continue_episode = 0