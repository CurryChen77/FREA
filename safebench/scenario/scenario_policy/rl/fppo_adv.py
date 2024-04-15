#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：fppo_adv.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/2/21
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""

import torch
import torch.nn as nn

from safebench.util.torch_util import CUDA
from safebench.scenario.scenario_policy.rl.ppo import PPO


class FPPOAdv(PPO):
    name = 'fppo_adv'
    type = 'onpolicy'

    def __init__(self, config, logger):
        super(FPPOAdv, self).__init__(config, logger)

    def set_feasibility_policy(self, feasibility_policy):
        self.feability_policy = feasibility_policy

    def get_feasibility_advantage_GAE(self, feasibility_V, feasibility_Q, undones):
        """
            unterminated: if the CBV collide with an object, then it is terminated
            undone: if the CBV is stuck or collide or max step will cause 'done'
            https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl/agents/AgentPPO.py
        """
        advantages = torch.empty_like(feasibility_V)  # advantage value

        horizon_len = feasibility_V.shape[0]

        advantage = torch.zeros_like(feasibility_V[0])  # last advantage value by GAE (Generalized Advantage Estimate)

        deltas = feasibility_V - feasibility_Q  # in feasibility, the lower, the better

        for t in range(horizon_len - 1, -1, -1):
            advantages[t] = advantage = deltas[t] + undones[t] * self.gamma * self.lambda_gae_adv * advantage
        return advantages

    def get_feasibility_Vs_Qs(self, closest_CBV_flag, ego_actions, ego_obs):
        feasibility_Vs = torch.full_like(closest_CBV_flag, -1.0)
        feasibility_Qs = torch.full_like(closest_CBV_flag, -1.0)
        # only consider the CBV is the closest BV from ego
        indices = torch.where(closest_CBV_flag > 0.5)[0]
        if indices.numel() > 0:
            action_inputs = ego_actions[indices]
            obs_inputs = ego_obs[indices]

            feasibility_Vs[indices] = self.feability_policy.get_feasibility_Vs(obs_inputs).squeeze()
            feasibility_Qs[indices] = self.feability_policy.get_feasibility_Qs(obs_inputs, action_inputs).squeeze()
        return feasibility_Vs, feasibility_Qs

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
            undones = CUDA(torch.FloatTensor(1-batch['dones']))
            unterminated = CUDA(torch.FloatTensor(1-batch['terminated']))
            buffer_size = states.shape[0]
            # feasibility
            closest_CBV_flag = CUDA(torch.FloatTensor(batch['closest_CBV_flag']))
            ego_actions = CUDA(torch.FloatTensor(batch['ego_actions']))
            ego_obs = CUDA(torch.FloatTensor(batch['ego_obs']))
            feasibility_Vs, feasibility_Qs = self.get_feasibility_Vs_Qs(closest_CBV_flag, ego_actions, ego_obs)

            # the values of the reward
            values = self.value(states)
            next_values = self.value(next_states)
            # the advantage of the reward
            reward_advantages = self.get_advantages_GAE(rewards, undones, values, next_values, unterminated)

            reward_sums = reward_advantages + values
            del rewards, values, next_values, unterminated, closest_CBV_flag, ego_obs, ego_actions

            # the advantage of the feasibility
            feasibility_advantages = self.get_feasibility_advantage_GAE(feasibility_Vs, feasibility_Qs, undones)

            # condition
            unsafe_condition = torch.where(feasibility_Vs > 0.0, 1.0, 0.0)
            safe_condition = torch.where(feasibility_Vs <= 0.0, 1.0, 0.0)

            # norm the reward advantage
            reward_advantages = (reward_advantages - reward_advantages.mean()) / (reward_advantages.std(dim=0) + 1e-5)
            # norm the feasibility advantage
            feasibility_advantages = (feasibility_advantages - feasibility_advantages.mean()) / (feasibility_advantages.std(dim=0) + 1e-5)

            # final advantage
            advantages = unsafe_condition * feasibility_advantages + safe_condition * reward_advantages

            del feasibility_Vs, feasibility_Qs, feasibility_advantages, reward_advantages, unsafe_condition, safe_condition, undones

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

