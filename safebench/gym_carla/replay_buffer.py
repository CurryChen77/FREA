#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：replay_buffer.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""

import numpy as np
import torch


class RouteReplayBuffer:
    """
        This buffer supports parallel storing transitions from multiple trajectories.
    """
    
    def __init__(self, num_scenario, mode, agent_config, buffer_capacity=1000):
        self.mode = mode
        self.buffer_capacity = buffer_capacity
        self.num_scenario = num_scenario
        self.buffer_len = 0
        self.need_obs = True if agent_config['obs_type'] == 'simple_state' or agent_config['obs_type'] == 'plant' else False

        # buffers for step info
        self.reset_buffer()

        # buffers for init info
        self.reset_init_buffer()

    def reset_buffer(self):
        self.buffer_ego_actions = [[] for _ in range(self.num_scenario)]
        self.buffer_scenario_actions = [[] for _ in range(self.num_scenario)]
        self.buffer_obs = [[] for _ in range(self.num_scenario)]
        self.buffer_next_obs = [[] for _ in range(self.num_scenario)]
        self.buffer_rewards = [[] for _ in range(self.num_scenario)]
        self.buffer_dones = [[] for _ in range(self.num_scenario)]
        self.buffer_additional_dict = [{} for _ in range(self.num_scenario)]

    def reset_init_buffer(self):
        self.buffer_static_obs = []
        self.buffer_init_action = []
        self.buffer_episode_reward = []
        self.buffer_init_additional_dict = {}
        self.init_buffer_len = 0

    def finish_one_episode(self):
        # get total reward for episode
        for s_i in range(self.num_scenario):
            dones = np.where(self.buffer_dones[s_i])[0]
            start_ = dones[-2] if len(dones) > 1 else -1
            end_ = dones[-1]
            self.buffer_episode_reward.append(np.sum(self.buffer_rewards[s_i][start_+1:end_+1]))

    def store(self, data_list, additional_dict):
        ego_actions = data_list[0]
        scenario_actions = data_list[1]
        obs = data_list[2]
        next_obs = data_list[3]
        rewards = data_list[4]
        dones = data_list[5]
        self.buffer_len += len(rewards)

        # separate trajectories according to infos
        for s_i in range(len(additional_dict)):
            sid = additional_dict[s_i]['scenario_id']
            self.buffer_ego_actions[sid].append(ego_actions[s_i])
            self.buffer_scenario_actions[sid].append(scenario_actions[s_i])
            if self.need_obs:
                self.buffer_obs[sid].append(obs[s_i])
                self.buffer_next_obs[sid].append(next_obs[s_i])
            self.buffer_rewards[sid].append(rewards[s_i])
            self.buffer_dones[sid].append(dones[s_i])

            # store additional information in given dict (e.g., collision and actor_info)
            for key in additional_dict[s_i].keys():
                if key == 'route_waypoints' or key == 'gps_route' or key == 'route':
                    continue
                if key not in self.buffer_additional_dict[s_i].keys():
                    self.buffer_additional_dict[s_i][key] = []
                self.buffer_additional_dict[s_i][key].append(additional_dict[s_i][key])

    def store_init(self, data_list, additional_dict=None):
        static_obs = data_list[0]
        scenario_init_action = data_list[1]
        self.buffer_static_obs.append(static_obs)
        self.buffer_init_action.append(scenario_init_action)
        self.init_buffer_len += len(scenario_init_action)

        # store additional information in given dict
        if additional_dict:
            for key in additional_dict.keys():
                if key not in self.buffer_init_additional_dict.keys():
                    self.buffer_init_additional_dict[key] = []
                self.buffer_init_additional_dict[key].append(additional_dict[key])

    def sample_init(self, batch_size):
        num_trajectory = len(self.buffer_init_action)
        start_idx = np.max([0, num_trajectory - self.buffer_capacity]) 

        # select up-to-date samples from buffer
        prepared_static_obs = self.buffer_static_obs[start_idx:]
        prepared_init_action = self.buffer_init_action[start_idx:]
        prepared_episode_reward = self.buffer_episode_reward[start_idx:]

        # sample action and episode reward
        sample_index = np.random.randint(0, len(prepared_init_action), size=batch_size)
        static_obs = np.concatenate(prepared_static_obs, axis=0)[sample_index]
        init_action = np.concatenate(prepared_init_action, axis=0)[sample_index]
        episode_reward = np.array(prepared_episode_reward)[sample_index]
        batch = {
            'static_obs': static_obs,
            'init_action': init_action,
            'episode_reward': episode_reward,
        }

        # add additional information to the batch (assume with torch)
        for key in self.buffer_init_additional_dict.keys():
            batch[key] = torch.cat(self.buffer_init_additional_dict[key][start_idx:])[sample_index]
        return batch

    def sample(self, batch_size):
        # prepare concatenated list
        prepared_ego_actions = []
        prepared_scenario_actions = []
        prepared_obs = []
        prepared_next_obs = []
        prepared_rewards = []
        prepared_dones = []
        prepared_infos = {}

        # get the length of each sub-buffer
        samples_per_trajectory = self.buffer_capacity // self.num_scenario  # assume average over all sub-buffer
        for s_i in range(self.num_scenario):
            # select the latest samples starting from the end of buffer
            num_trajectory = len(self.buffer_rewards[s_i])
            start_idx = np.max([0, num_trajectory - samples_per_trajectory])

            # concat
            prepared_ego_actions += self.buffer_ego_actions[s_i][start_idx:]
            prepared_scenario_actions += self.buffer_scenario_actions[s_i][start_idx:]
            if self.need_obs:
                prepared_obs += self.buffer_obs[s_i][start_idx:]
                prepared_next_obs += self.buffer_next_obs[s_i][start_idx:]
            prepared_rewards += self.buffer_rewards[s_i][start_idx:]
            prepared_dones += self.buffer_dones[s_i][start_idx:]

            # add additional information 
            for k_i in self.buffer_additional_dict[s_i].keys():
                if k_i not in prepared_infos.keys():
                    prepared_infos[k_i] = []
                prepared_infos[k_i] += self.buffer_additional_dict[s_i][k_i][start_idx:]

        # sample from concatenated list
        # the first sample does not have previous state ()
        sample_index = np.random.randint(1, len(prepared_rewards), size=batch_size)

        # prepare batch 
        action = prepared_ego_actions if self.mode == 'train_agent' else prepared_scenario_actions
        batch = {
            'action': np.stack(action)[sample_index],                 # action
            'reward': np.stack(prepared_rewards)[sample_index],       # reward
            'done': np.stack(prepared_dones)[sample_index],           # done
        }
        if self.need_obs:
            batch.update({
                'state': np.stack(prepared_obs)[sample_index, :],         # state
                'n_state': np.stack(prepared_next_obs)[sample_index, :],  # next state
            })

        # add additional information to the batch
        batch_info = {} 
        for k_i in prepared_infos.keys():
            if k_i == 'route_waypoints' or k_i == 'gps_route' or k_i == 'route':
                continue
            batch_info[k_i] = np.stack(prepared_infos[k_i])[sample_index-1]
            batch_info['n_' + k_i] = np.stack(prepared_infos[k_i])[sample_index]

        # combine two dicts
        batch.update(batch_info)
        return batch
