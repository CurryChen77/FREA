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
from copy import deepcopy

from frea.util.torch_util import CUDA
from frea.scenario.scenario_policy.rl.ppo import PPO


class FPPOAdv(PPO):
    feasibility_min_dis_threshold: float
    name = 'fppo_adv'
    type = 'onpolicy'

    def __init__(self, config, logger):
        super(FPPOAdv, self).__init__(config, logger)

    def set_feasibility_policy(self, feasibility_policy):
        self.feability_policy = feasibility_policy
        self.feasibility_M = self.feability_policy.M
        self.feasibility_min_dis_threshold = self.feability_policy.min_dis_threshold

    def get_feasibility_advantage_GAE(self, feasibility_V, feasibility_next_V, ego_CBV_dis, ego_CBV_next_dis, undones):
        """
            unterminated: if the CBV collide with an object, then it is terminated
            undone: if the CBV is stuck or collide or max step will cause 'done'
            https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl/agents/AgentPPO.py
        """
        advantages = torch.empty_like(feasibility_V)  # advantage value

        horizon_len = feasibility_V.shape[0]

        advantage = torch.zeros_like(feasibility_V[0])  # last advantage value by GAE (Generalized Advantage Estimate)

        # h(s') < h(s) <-> next_ego_min_dis > self.feasibility_min_dis_threshold and ego_min_dis <= self.feasibility_min_dis_threshold
        condition = (ego_CBV_next_dis > self.feasibility_min_dis_threshold) & (ego_CBV_dis <= self.feasibility_min_dis_threshold)
        # if h(s') < h(s), feasibility_Qh = max(h(s), Vh(s'))
        # elif h(s') >= h(s), feasibility_Qh = Vh(s')
        feasibility_Q = deepcopy(feasibility_next_V)
        feasibility_Q[condition] = torch.maximum(feasibility_next_V[condition], CUDA(torch.tensor(self.feasibility_M)))

        deltas = feasibility_V - feasibility_Q  # feasibility_V > feasibility_Q means the next state is much safer

        for t in range(horizon_len - 1, -1, -1):
            advantages[t] = advantage = deltas[t] + undones[t] * self.gamma * self.lambda_gae_adv * advantage
        return advantages

    def get_feasibility_Vs(self, ego_CBV_obs, ego_CBV_next_obs):
        assert len(ego_CBV_obs) == len(ego_CBV_next_obs)
        length = len(ego_CBV_obs)
        # calculate the feasibility_V
        feasibility_all_V = self.feability_policy.get_feasibility_Vs(torch.cat((ego_CBV_obs, ego_CBV_next_obs), dim=0))
        feasibility_V = feasibility_all_V[:length]
        feasibility_next_V = feasibility_all_V[length:]

        return [feasibility_V, feasibility_next_V]

    def get_surrogate_advantages(self, feasibility_advantages, reward_advantages, feasibility_V, feasibility_next_V):
        # current safe condition
        safe_condition = torch.where(feasibility_V <= 0.0, 1.0, 0.0)
        unsafe_condition = torch.where(feasibility_V > 0.0, 1.0, 0.0)
        # next safe condition
        next_safe_condition = torch.where(feasibility_next_V <= 0.0, 1.0, 0.0)
        next_unsafe_condition = torch.where(feasibility_next_V > 0.0, 1.0, 0.0)

        # final advantage
        advantages = unsafe_condition * feasibility_advantages + \
                    safe_condition * (next_safe_condition * reward_advantages + next_unsafe_condition * feasibility_advantages)

        del safe_condition, unsafe_condition, next_safe_condition, next_unsafe_condition

        return advantages

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
            ego_CBV_obs = CUDA(torch.FloatTensor(batch['ego_CBV_obs']))
            ego_CBV_next_obs = CUDA(torch.FloatTensor(batch['ego_CBV_next_obs']))
            ego_CBV_dis = CUDA(torch.FloatTensor(batch['ego_CBV_dis']))
            ego_CBV_next_dis = CUDA(torch.FloatTensor(batch['ego_CBV_next_dis']))
            feasibility_V, feasibility_next_V = self.get_feasibility_Vs(ego_CBV_obs, ego_CBV_next_obs)
            writer.add_scalar("unsafe ratio", (feasibility_next_V > 0).float().mean().item(), e_i)

            # the values of the reward
            values = self.value(states)
            next_values = self.value(next_states)
            # the advantage of the reward
            reward_advantages = self.get_advantages_GAE(rewards, undones, values, next_values, unterminated)

            reward_sums = reward_advantages + values
            del rewards, values, next_values, unterminated, ego_CBV_obs, ego_CBV_next_obs

            # the advantage of the feasibility
            feasibility_advantages = self.get_feasibility_advantage_GAE(feasibility_V, feasibility_next_V, ego_CBV_dis, ego_CBV_next_dis, undones)

            # norm the advantage
            reward_advantages = (reward_advantages - reward_advantages.mean()) / (reward_advantages.std(dim=0) + 1e-5)
            feasibility_advantages = (feasibility_advantages - feasibility_advantages.mean()) / (feasibility_advantages.std(dim=0) + 1e-5)

            # the surrogate_advantages combining safe and unsafe conditions
            advantages = self.get_surrogate_advantages(feasibility_advantages, reward_advantages, feasibility_V, feasibility_next_V)

            del feasibility_V, feasibility_next_V, feasibility_advantages, reward_advantages, undones, ego_CBV_dis, ego_CBV_next_dis

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

