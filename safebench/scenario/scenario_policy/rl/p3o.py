#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：fppo_adv.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/2/21
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""
import os
from fnmatch import fnmatch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from safebench.util.torch_util import CUDA
from safebench.scenario.scenario_policy.rl.ppo import PPO
from safebench.gym_carla.net import ActorPPO, CriticPPO


class P3O(PPO):
    feasibility_min_dis_threshold: float
    name = 'p3o'
    type = 'onpolicy'

    def __init__(self, config, logger):
        super(P3O, self).__init__(config, logger)
        self.value_c_lr = config['value_c_lr']
        self.value_c = CUDA(CriticPPO(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))
        self.value_c_optim = torch.optim.Adam(self.value.parameters(), lr=self.value_lr, eps=1e-5)  # trick about eps
        self.value_c_criterion = nn.SmoothL1Loss()
        self.kappa = config['kappa']
        self.cost_limit = config['cost_limit']

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.policy.train()
            self.value.train()
            self.value_c.train()
        elif mode == 'eval':
            self.policy.eval()
            self.value.eval()
            self.value_c.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')

    def lr_decay(self, e_i):
        lr_policy_now = self.policy_lr * (1 - e_i / self.map_train_episode)
        lr_value_now = self.value_lr * (1 - e_i / self.map_train_episode)
        lr_value_c_now = self.value_c_lr * (1 - e_i / self.map_train_episode)
        for p in self.policy_optim.param_groups:
            p['lr'] = lr_policy_now
        for p in self.value_optim.param_groups:
            p['lr'] = lr_value_now
        for p in self.value_c_optim.param_groups:
            p['lr'] = lr_value_c_now

    def set_feasibility_policy(self, feasibility_policy):
        self.feability_policy = feasibility_policy
        self.feasibility_M = self.feability_policy.M
        self.feasibility_min_dis_threshold = self.feability_policy.min_dis_threshold

    def get_feasibility_cost(self, next_closest_CBV_flag, ego_next_obs):
        feasibility_next_V = torch.full_like(next_closest_CBV_flag, -1.0)
        feasibility_cost = torch.full_like(next_closest_CBV_flag, 0.0)
        # only consider the CBV is the closest BV from ego
        indices = next_closest_CBV_flag > 0.5

        if indices.any():
            # calculate the feasibility_next_V
            feasibility_next_V[indices] = self.feability_policy.get_feasibility_Vs(ego_next_obs[indices])
            feasibility_cost.masked_fill_(feasibility_next_V > 0.0, 1.0)

        return feasibility_cost

    def get_EpCost(self, feasibility_costs, dones):
        episode_costs = []
        episode_sum = 0

        for i in range(len(dones)):
            episode_sum += feasibility_costs[i]

            if dones[i] > 0.5:
                episode_costs.append(episode_sum)
                episode_sum = 0

        if dones[-1] < 0.5:
            episode_costs.append(episode_sum)

        EpCost = torch.tensor(episode_costs).mean().item() if episode_costs else 0
        return EpCost

    def loss_pi(self, state, action, log_prob, reward_adv):
        new_log_prob, entropy = self.policy.get_logprob_entropy(state, action)

        ratio = (new_log_prob - log_prob.detach()).exp()
        L1 = reward_adv * ratio
        L2 = reward_adv * torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
        surrogate = torch.min(L1, L2).mean()
        loss_reward = -(surrogate + entropy.mean() * self.lambda_entropy)

        return loss_reward.mean()

    def loss_pi_cost(self, state, action, log_prob, cost_adv, EpCost):
        new_log_prob, entropy = self.policy.get_logprob_entropy(state, action)

        ratio = (new_log_prob - log_prob.detach()).exp()
        surr_cost_adv = (cost_adv * ratio).mean()
        Jc = EpCost - self.cost_limit

        loss_cost = self.kappa * F.relu(surr_cost_adv + Jc)

        return loss_cost.mean()

    def train(self, buffer, writer, e_i):
        with torch.no_grad():
            # learning rate decay
            self.lr_decay(e_i)  # add the learning rate decay

            batch = buffer.get()

            states = CUDA(torch.FloatTensor(batch['obs']))
            next_states = CUDA(torch.FloatTensor(batch['next_obs']))
            actions = CUDA(torch.FloatTensor(batch['actions']))
            log_probs = CUDA(torch.FloatTensor(batch['log_probs']))
            rewards = CUDA(torch.FloatTensor(batch['rewards']))
            dones = CUDA(torch.FloatTensor(batch['dones']))
            undones = CUDA(torch.FloatTensor(1-batch['dones']))
            unterminated = CUDA(torch.FloatTensor(1-batch['terminated']))
            buffer_size = states.shape[0]
            # feasibility
            next_closest_CBV_flag = CUDA(torch.FloatTensor(batch['next_closest_CBV_flag']))
            ego_next_obs = CUDA(torch.FloatTensor(batch['ego_next_obs']))

            # the values of the reward
            values = self.value(states)
            next_values = self.value(next_states)
            # the advantage of the reward
            reward_advantages = self.get_advantages_GAE(rewards, undones, values, next_values, unterminated)

            # the value of the cost
            feasibility_cost = self.get_feasibility_cost(next_closest_CBV_flag, ego_next_obs)
            EpCost = self.get_EpCost(feasibility_cost, dones)
            writer.add_scalar("Episode average cost", EpCost, e_i)
            values_c = self.value_c(states)
            next_values_c = self.value_c(next_states)
            # the advantage of the cost
            cost_advantages = self.get_advantages_GAE(feasibility_cost, undones, values_c, next_values_c, unterminated)

            reward_sums = reward_advantages + values
            cost_sums = cost_advantages + values_c
            del rewards, values, next_values, values_c, next_values_c, feasibility_cost, unterminated, next_closest_CBV_flag, undones, dones

            # norm the advantage
            reward_advantages = (reward_advantages - reward_advantages.mean()) / (reward_advantages.std(dim=0) + 1e-5)
            cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std(dim=0) + 1e-5)

        # start to train, use gradient descent without batch_size
        update_times = int(buffer_size * self.train_repeat_times / self.batch_size)
        assert update_times >= 1

        for _ in range(update_times):
            indices = torch.randint(buffer_size, size=(self.batch_size,), requires_grad=False)
            state = states[indices]
            action = actions[indices]
            log_prob = log_probs[indices]
            reward_advantage = reward_advantages[indices]
            cost_advantage = cost_advantages[indices]
            reward_sum = reward_sums[indices]
            cost_sum = cost_sums[indices]

            # update value function
            value = self.value(state)
            value_loss = self.value_criterion(value, reward_sum)  # the value criterion is SmoothL1Loss() instead of MSE
            self.value_optim.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
            self.value_optim.step()

            # update cost value function
            value_c = self.value_c(state)
            value_c_loss = self.value_c_criterion(value_c, cost_sum)  # the value criterion is SmoothL1Loss() instead of MSE
            self.value_c_optim.zero_grad()
            value_c_loss.backward()
            nn.utils.clip_grad_norm_(self.value_c.parameters(), 0.5)
            self.value_c_optim.step()

            # update policy
            loss_reward = self.loss_pi(state, action, log_prob, reward_advantage)
            loss_cost = self.loss_pi_cost(state, action, log_prob, cost_advantage, EpCost)
            actor_loss = loss_reward + loss_cost
            self.policy_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optim.step()

            writer.add_scalar("reward value net loss", value_loss.item(), e_i)
            writer.add_scalar("cost value net loss", value_c_loss.item(), e_i)
            writer.add_scalar("loss_reward", loss_reward.item(), e_i)
            writer.add_scalar("loss_cost", loss_cost.item(), e_i)

        # reset buffer
        buffer.reset_buffer()

    def save_model(self, episode, map_name, buffer):
        states = {
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'value_c': self.value_c.state_dict(),
            'policy_optim': self.policy_optim.state_dict(),
            'value_optim': self.value_optim.state_dict(),
            'value_c_optim': self.value_c_optim.state_dict()
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
            self.logger.log(f'>> Loading scenario policy {self.name} model from {os.path.basename(filepath)}', 'yellow')
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.policy.load_state_dict(checkpoint['policy'])
            self.value.load_state_dict(checkpoint['value'])
            self.value_c.load_state_dict(checkpoint['value_c'])
            self.policy_optim.load_state_dict(checkpoint['policy_optim'])
            self.value_optim.load_state_dict(checkpoint['value_optim'])
            self.value_c_optim.load_state_dict(checkpoint['value_c_optim'])
            self.continue_episode = episode
        else:
            self.logger.log(f'>> No scenario policy {self.name} model found at {filepath}', 'red')
            self.policy = CUDA(ActorPPO(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))
            self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr, eps=1e-5)  # trick about eps
            self.value = CUDA(CriticPPO(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))
            self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.value_lr, eps=1e-5)  # trick about eps
            self.value_c = CUDA(CriticPPO(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))
            self.value_c_optim = torch.optim.Adam(self.value.parameters(), lr=self.value_lr, eps=1e-5)  # trick about eps
            self.continue_episode = 0