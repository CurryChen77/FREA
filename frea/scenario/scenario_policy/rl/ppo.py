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

from frea.util.torch_util import CUDA, CPU
from frea.scenario.scenario_policy.base_policy import BasePolicy
from frea.gym_carla.net import ActorPPO, CriticPPO


class PPO(BasePolicy):
    name = 'ppo'
    type = 'onpolicy'

    def __init__(self, config, logger):
        super(PPO, self).__init__(config, logger)

        self.continue_episode = 0
        self.logger = logger
        self.gamma = config['gamma']
        self.policy_lr = config['policy_lr']
        self.value_lr = config['value_lr']
        self.seed = config['seed']
        self.train_repeat_times = config['train_repeat_times']

        self.state_dim = config['scenario_state_dim']
        self.action_dim = config['scenario_action_dim']
        self.clip_epsilon = config['clip_epsilon']
        self.batch_size = config['batch_size']
        self.lambda_gae_adv = config['lambda_gae_adv']
        self.lambda_entropy = config['lambda_entropy']
        self.lambda_entropy = CUDA(torch.tensor(self.lambda_entropy, dtype=torch.float32))
        self.train_episode_list = config['train_episode']
        self.map_train_episode = None
        self.dims = config['dims']

        self.model_type = config['model_type']
        self.CBV_selection = config['CBV_selection']
        self.model_path = os.path.join(config['ROOT_DIR'], config['model_path'])
        self.scenario_id = config['scenario_id']
        if config['mode'] == 'eval':
            self.agent_info = config['pretrain_ego'] + '_' + str(self.CBV_selection) + '_seed' + str(self.seed)
        else:
            self.agent_info = config['agent_policy'] + '_' + str(self.CBV_selection) + '_seed' + str(self.seed)

        self.policy = CUDA(ActorPPO(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr, eps=1e-5)  # trick about eps
        self.value = CUDA(CriticPPO(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.value_lr, eps=1e-5)  # trick about eps
        self.value_criterion = nn.SmoothL1Loss()

        self.mode = 'train'

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.policy.train()
            self.value.train()
        elif mode == 'eval':
            self.policy.eval()
            self.value.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')

    def lr_decay(self, e_i):
        lr_policy_now = self.policy_lr * (1 - e_i / self.map_train_episode)
        lr_value_now = self.value_lr * (1 - e_i / self.map_train_episode)
        for p in self.policy_optim.param_groups:
            p['lr'] = lr_policy_now
        for p in self.value_optim.param_groups:
            p['lr'] = lr_value_now

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

    def get_action(self, state, infos, deterministic=False):
        state, CBVs_id, env_index = self.info_process(infos)
        scenario_action = [{} for _ in range(len(infos))]
        scenario_log_prob = [{} for _ in range(len(infos))]
        if state is not None:
            state_tensor = CUDA(torch.FloatTensor(state))
            if deterministic:
                action = self.policy(state_tensor)
                for i, (CBV_id, env_id) in enumerate(zip(CBVs_id, env_index)):
                    scenario_action[env_id][CBV_id] = CPU(action[i])
            else:
                action, log_prob = self.policy.get_action(state_tensor)
                for i, (CBV_id, env_id) in enumerate(zip(CBVs_id, env_index)):
                    scenario_action[env_id][CBV_id] = CPU(action[i])
                    scenario_log_prob[env_id][CBV_id] = CPU(log_prob[i])
        return scenario_action, scenario_log_prob

    def get_advantages_GAE(self, rewards, undones, values, next_values, unterminated):
        """
            unterminated: if the CBV collide with an object, then it is terminated
            undone: if the CBV is stuck or collide or max step will cause 'done'
            https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl/agents/AgentPPO.py
        """
        advantages = torch.empty_like(values)  # advantage value

        horizon_len = rewards.shape[0]

        advantage = torch.zeros_like(values[0])  # last advantage value by GAE (Generalized Advantage Estimate)

        for t in range(horizon_len - 1, -1, -1):
            delta = rewards[t] + unterminated[t] * self.gamma * next_values[t] - values[t]
            advantages[t] = advantage = delta + undones[t] * self.gamma * self.lambda_gae_adv * advantage
        return advantages

    def train(self, buffer, writer, e_i):
        """
            from https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl/agents/AgentPPO.py
        """

        with torch.no_grad():
            # learning rate decay
            self.lr_decay(e_i)  # add the learning rate decay

            batch = buffer.get()

            states = CUDA(torch.FloatTensor(batch['obs']))
            next_states = CUDA(torch.FloatTensor(batch['next_obs']))
            actions = CUDA(torch.FloatTensor(batch['actions']))
            log_probs = CUDA(torch.FloatTensor(batch['log_probs']))
            rewards = CUDA(torch.FloatTensor(batch['rewards']))
            undones = CUDA(torch.FloatTensor(1-batch['dones']))
            unterminated = CUDA(torch.FloatTensor(1-batch['terminated']))
            buffer_size = states.shape[0]

            values = self.value(states)
            next_values = self.value(next_states)

            advantages = self.get_advantages_GAE(rewards, undones, values, next_values, unterminated)
            reward_sums = advantages + values
            del rewards, undones, values, next_values, unterminated

            advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-5)

        # start to train, use gradient descent without batch size
        update_times = int(buffer_size * self.train_repeat_times / self.batch_size)
        assert update_times >= 1

        for _ in range(update_times):
            indices = torch.randint(buffer_size, size=(self.batch_size,), requires_grad=False)
            state = states[indices]
            action = actions[indices]
            log_prob = log_probs[indices]
            advantage = advantages[indices]
            reward_sum = reward_sums[indices]

            # update value function
            value = self.value(state)
            value_loss = self.value_criterion(value, reward_sum)  # the value criterion is SmoothL1Loss() instead of MSE
            writer.add_scalar("value loss", value_loss, e_i)
            self.value_optim.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
            self.value_optim.step()

            # update policy
            new_log_prob, entropy = self.policy.get_logprob_entropy(state, action)

            ratio = (new_log_prob - log_prob.detach()).exp()
            L1 = advantage * ratio
            L2 = advantage * torch.clamp(ratio, 1.0-self.clip_epsilon, 1.0+self.clip_epsilon)
            surrogate = torch.min(L1, L2).mean()
            actor_loss = -(surrogate + entropy.mean() * self.lambda_entropy)
            writer.add_scalar("actor entropy", entropy.mean(), e_i)
            writer.add_scalar("actor loss", actor_loss, e_i)
            self.policy_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optim.step()

        # reset buffer
        buffer.reset_buffer()

    def save_model(self, episode, map_name, buffer):
        states = {
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'policy_optim': self.policy_optim.state_dict(),
            'value_optim': self.value_optim.state_dict()
        }
        scenario_name = "all" if self.scenario_id is None else 'Scenario' + str(self.scenario_id)
        save_dir = os.path.join(self.model_path, self.agent_info, scenario_name+"_"+map_name)
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f'model.{self.name}.{self.model_type}.{episode:04}.torch')
        self.logger.log(f'>> Saving scenario policy {self.name} model to {os.path.basename(filepath)}', 'yellow')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)

    def load_model(self, map_name, episode=None):
        self.map_train_episode = self.train_episode_list[map_name]
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
        filepath = os.path.join(load_dir, f'model.{self.name}.{self.model_type}.{episode:04}.torch')
        if os.path.isfile(filepath):
            self.logger.log(f'>> Loading scenario policy {self.name} model from {self.agent_info}', 'yellow')
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.policy.load_state_dict(checkpoint['policy'])
            self.value.load_state_dict(checkpoint['value'])
            self.policy_optim.load_state_dict(checkpoint['policy_optim'])
            self.value_optim.load_state_dict(checkpoint['value_optim'])
            self.continue_episode = episode
        else:
            self.logger.log(f'>> No scenario policy {self.name} model found at {filepath}', 'red')
            self.policy = CUDA(ActorPPO(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))
            self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr, eps=1e-5)  # trick about eps
            self.value = CUDA(CriticPPO(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))
            self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.value_lr, eps=1e-5)  # trick about eps
            self.continue_episode = 0