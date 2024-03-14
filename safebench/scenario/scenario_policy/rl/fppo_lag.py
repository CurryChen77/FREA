#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：fppo_lag
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/3/8
"""
import os

import numpy as np
import torch

import torch.nn as nn
from fnmatch import fnmatch

from safebench.util.torch_util import CUDA

from safebench.gym_carla.Lagrange import Lagrange
from safebench.scenario.scenario_policy.rl.ppo import PPO


class FPPOLag(PPO):
    name = 'fppo_lag'
    type = 'onpolicy'

    def __init__(self, config, logger):
        super(FPPOLag, self).__init__(config, logger)
        self.lagrange = Lagrange(**config['lagrange_cfgs'])
        self.constraint_upper_bound = config['constraint_upper_bound']

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.policy.train()
            self.value.train()
            self.lagrange.lagrangian_multiplier.train()
        elif mode == 'eval':
            self.policy.eval()
            self.value.eval()
            self.lagrange.lagrangian_multiplier.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')

    def lr_decay(self, e_i):
        lr_policy_now = self.policy_lr * (1 - e_i / self.max_train_episode)
        lr_value_now = self.value_lr * (1 - e_i / self.max_train_episode)
        lr_lagrange_multiplier_now = self.lagrange.lambda_lr * (1 - e_i / self.max_train_episode)
        for p in self.policy_optim.param_groups:
            p['lr'] = lr_policy_now
        for p in self.value_optim.param_groups:
            p['lr'] = lr_value_now
        for p in self.lagrange.lambda_optimizer.param_groups:
            p['lr'] = lr_lagrange_multiplier_now

    def train(self, buffer, writer, e_i):
        """
            from https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/helloworld/helloworld_PPO_single_file.py#L29
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
            feasibility_Qs = CUDA(torch.FloatTensor(batch['feasibility_Qs']))
            feasibility_Vs = CUDA(torch.FloatTensor(batch['feasibility_Vs']))
            buffer_size = states.shape[0]

            # the values of the reward
            values = self.value(states)
            next_values = self.value(next_states)
            # the advantage of the reward
            reward_advantages = self.get_advantages_vtrace(rewards, undones, values, next_values, unterminated)
            reward_sums = reward_advantages + values
            del rewards, undones, values, next_values, unterminated

            # the advantage of the feasibility
            feasibility_advantages = feasibility_Qs - feasibility_Vs

            # Lagrange multiplier
            penalty = self.lagrange.get_lagrangian_multiplier(states)
            constraints = torch.clamp(feasibility_Vs - self.constraint_upper_bound, min=-5., max=30)

            # final advantage
            advantages = (reward_advantages - penalty * feasibility_advantages) / (1 + penalty)
            # norm the final advantage
            advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-5)
            del feasibility_Vs, feasibility_Qs, feasibility_advantages, reward_advantages, penalty

        # start to train, use gradient descent without batch_size
        update_times = int(buffer_size * self.train_repeat_times / self.batch_size)
        assert update_times >= 1

        for _ in range(update_times):
            indices = torch.randint(buffer_size, size=(self.batch_size,), requires_grad=False)
            state = states[indices]
            action = actions[indices]
            log_prob = log_probs[indices]
            advantage = advantages[indices]
            reward_sum = reward_sums[indices]
            constraint = constraints[indices]

            # update the Lagrange multipliers
            lambda_loss = self.lagrange.update_lagrange_multiplier(state, constraint)
            writer.add_scalar("lambda loss", lambda_loss, e_i)

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
            'value_optim': self.value_optim.state_dict(),
            'lagrange_multiplier': self.lagrange.lagrangian_multiplier.state_dict(),
            'lagrange_multiplier_optim': self.lagrange.lambda_optimizer.state_dict(),
        }
        scenario_name = "all" if self.scenario_id is None else 'Scenario' + str(self.scenario_id)
        save_dir = os.path.join(self.model_path, self.agent_info, scenario_name+"_"+map_name)
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f'model.{self.name}.{self.model_type}.{episode:04}.torch')
        self.logger.log(f'>> Saving scenario policy {self.name} model to {os.path.basename(filepath)}', 'yellow')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)

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
        filepath = os.path.join(load_dir, f'model.{self.name}.{self.model_type}.{episode:04}.torch')
        if os.path.isfile(filepath):
            self.logger.log(f'>> Loading scenario policy {self.name} model from {os.path.basename(filepath)}', 'yellow')
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.policy.load_state_dict(checkpoint['policy'])
            self.value.load_state_dict(checkpoint['value'])
            self.policy_optim.load_state_dict(checkpoint['policy_optim'])
            self.value_optim.load_state_dict(checkpoint['value_optim'])
            self.lagrange.lagrangian_multiplier.load_state_dict(checkpoint['lagrange_multiplier'])
            self.lagrange.lambda_optimizer.load_state_dict(checkpoint['lagrange_multiplier_optim'])
            self.continue_episode = episode
        else:
            self.logger.log(f'>> No scenario policy {self.name} model found at {filepath}', 'red')
            self.continue_episode = 0
