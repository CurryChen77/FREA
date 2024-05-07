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

from frea.util.torch_util import CUDA, CPU
from frea.scenario.scenario_policy.rl.ppo import PPO


def process_feasibility_rewards(feasibility_next_V, clamp_range, map_range):
    clamp_min, clamp_max = clamp_range
    map_min, map_max = map_range

    mask = feasibility_next_V > 0

    output = torch.zeros_like(feasibility_next_V)

    clamped_and_mapped_V = ((torch.clamp(feasibility_next_V[mask], min=clamp_min, max=clamp_max) - clamp_min) / (clamp_max - clamp_min)) * (map_max - map_min) + map_min

    output[mask] = clamped_and_mapped_V

    return output


class FPPORs(PPO):
    name = 'fppo_rs'
    type = 'onpolicy'

    def __init__(self, config, logger):
        super(FPPORs, self).__init__(config, logger)
        self.reward_punish = config['reward_punish']

    def set_feasibility_policy(self, feasibility_policy):
        self.feability_policy = feasibility_policy

    def get_feasibility_rewards(self, ego_CBV_next_obs):
        feasibility_next_V = self.feability_policy.get_feasibility_Vs(ego_CBV_next_obs).squeeze()
        feasibility_rewards = -1 * process_feasibility_rewards(feasibility_next_V, clamp_range=(0, 8), map_range=(1, 2))
        return feasibility_rewards, feasibility_next_V

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
            # feasibility
            ego_CBV_next_obs = CUDA(torch.FloatTensor(batch['ego_CBV_next_obs']))
            feasibility_rewards, feasibility_next_V = self.get_feasibility_rewards(ego_CBV_next_obs)
            writer.add_scalar("unsafe ratio", (feasibility_next_V > 0).float().mean().item(), e_i)

            rewards += feasibility_rewards

            values = self.value(states)
            next_values = self.value(next_states)

            advantages = self.get_advantages_GAE(rewards, undones, values, next_values, unterminated)
            reward_sums = advantages + values
            del rewards, undones, values, next_values, unterminated, feasibility_rewards, ego_CBV_next_obs, feasibility_next_V

            advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-5)

        # start to train, use gradient descent without the batch size
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


