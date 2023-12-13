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
    
    def __init__(self, num_scenario, mode, agent_config=None, safety_network_config=None, buffer_capacity=10000):
        self.mode = mode
        self.buffer_capacity = buffer_capacity
        self.num_scenario = num_scenario
        self.agent_need_obs = True if (agent_config['obs_type'] == 'simple_state' or agent_config['obs_type'] == 'plant') else False

        self.safety_network_obs_type = safety_network_config['obs_type'] if safety_network_config else None
        self.pos = 0
        self.buffer_len = 0
        self.full = False

        # buffers for step info
        self.reset_buffer()

    def reset_buffer(self):
        if self.mode == 'train_scenario':
            self.buffer_actions = []
            self.buffer_obs = []
            self.buffer_next_obs = []
            self.buffer_rewards = []
            self.buffer_dones = []
        elif self.mode == 'train_agent':
            self.buffer_actions = []
            self.buffer_obs = []
            self.buffer_next_obs = []
            self.buffer_rewards = []
            self.buffer_dones = []
        elif self.mode == 'train_safety_network':
            self.buffer_next_obs = []
            self.buffer_constrain_h = []
            self.buffer_dones = []

    def store(self, data_list, additional_dict):
        """
            additional dict[0]: the current infos
            additional dict[1]: the next infos
        """
        # store for scenario training
        if self.mode == 'train_scenario':
            scenario_actions = data_list[1]
            dones = data_list[5]
            obs = [info['scenario_obs'] for info in additional_dict[0]]
            next_obs = [info['scenario_obs'] for info in additional_dict[1]]
            rewards = [info['scenario_agent_reward'] for info in additional_dict[1]]  # the scenario reward should from the next infos
            if not self.full:
                for s_i in range(len(dones)):
                    self.buffer_actions.append(scenario_actions[s_i])
                    self.buffer_obs.append(obs[s_i])  # cbv obs is scenario_obs
                    self.buffer_next_obs.append(next_obs[s_i])  # cbv next obs is the scenario_obs from next info
                    self.buffer_rewards.append(rewards[s_i])  # cbv reward is scenario_agent_reward
                    self.buffer_dones.append(dones[s_i])
                    self.pos += 1
                    if self.pos == self.buffer_capacity:
                        self.full = True
                        self.pos = 0
                        break
            else:
                for s_i in range(len(dones)):
                    self.buffer_actions[self.pos] = scenario_actions[s_i]
                    self.buffer_obs[self.pos] = obs[s_i]  # cbv obs is scenario_obs
                    self.buffer_next_obs[self.pos] = next_obs[s_i]  # cbv next obs is the scenario_obs from next info
                    self.buffer_rewards[self.pos] = rewards[s_i]  # cbv reward is scenario_agent_reward
                    self.buffer_dones[self.pos] = dones[s_i]
                    self.pos += 1
                    if self.pos == self.buffer_capacity:
                        self.pos = 0

        # store for agent training
        elif self.mode == 'train_agent':
            ego_actions = data_list[0]
            obs = data_list[2]
            next_obs = data_list[3]
            rewards = data_list[4]
            dones = data_list[5]
            if not self.full:
                for s_i in range(len(dones)):
                    self.buffer_actions.append(ego_actions[s_i])
                    self.buffer_obs.append(obs[s_i])
                    self.buffer_next_obs.append(next_obs[s_i])
                    self.buffer_rewards.append(rewards[s_i])
                    self.buffer_dones.append(dones[s_i])
                    self.pos += 1
                    if self.pos == self.buffer_capacity:
                        self.full = True
                        self.pos = 0
                        break
            else:
                for s_i in range(len(dones)):
                    self.buffer_actions[self.pos] = ego_actions[s_i]
                    self.buffer_obs[self.pos] = obs[s_i]
                    self.buffer_next_obs[self.pos] = next_obs[s_i]
                    self.buffer_rewards[self.pos] = rewards[s_i]
                    self.buffer_dones[self.pos] = dones[s_i]
                    self.pos += 1
                    if self.pos == self.buffer_capacity:
                        self.pos = 0

        # store for agent training
        elif self.mode == 'train_safety_network':
            dones = data_list[5]
            if self.safety_network_obs_type and self.safety_network_obs_type == 'encoded_state':
                next_obs = [info['encoded_state'] for info in additional_dict[1]]
            elif self.safety_network_obs_type and self.safety_network_obs_type == 'ego_info':
                next_obs = [info['ego_info'] for info in additional_dict[1]]
            else:
                raise ValueError(f'Unknown safety_network obs_type')
            constrain_h = [info['constrain_h'] for info in additional_dict[1]]  # the constraint h should from next infos
            if not self.full:
                for s_i in range(len(dones)):
                    self.buffer_next_obs.append(next_obs[s_i])
                    self.buffer_constrain_h.append(constrain_h[s_i])
                    self.buffer_dones.append(dones[s_i])
                    self.pos += 1
                    if self.pos == self.buffer_capacity:
                        self.full = True
                        self.pos = 0
                        break
            else:
                for s_i in range(len(dones)):
                    self.buffer_next_obs[self.pos] = next_obs[s_i]
                    self.buffer_constrain_h[self.pos] = constrain_h[s_i]
                    self.buffer_dones[self.pos] = dones[s_i]
                    self.pos += 1
                    if self.pos == self.buffer_capacity:
                        self.pos = 0

        self.buffer_len = len(self.buffer_dones)

    def sample(self, batch_size):
        upper_bound = self.buffer_capacity if self.full else self.pos
        batch_indices = np.random.randint(0, high=upper_bound, size=batch_size)

        if self.mode == 'train_scenario':
            batch = {
                'action': np.stack(self.buffer_actions)[batch_indices],                 # action
                'scenario_obs': np.stack(self.buffer_obs)[batch_indices, :],            # obs
                'next_scenario_obs': np.stack(self.buffer_next_obs)[batch_indices, :],  # next obs
                'reward': np.stack(self.buffer_rewards)[batch_indices],                 # reward
                'done': np.stack(self.buffer_dones)[batch_indices],                     # done
            }
        elif self.mode == 'train_agent':
            batch = {
                'action': np.stack(self.buffer_actions)[batch_indices],                    # action
                'obs': np.stack(self.buffer_obs)[batch_indices, :],                        # obs
                'next_obs': np.stack(self.buffer_next_obs)[batch_indices, :],              # next obs
                'reward': np.stack(self.buffer_rewards)[batch_indices],                 # reward
                'done': np.stack(self.buffer_dones)[batch_indices],                     # done
            }
        elif self.mode == 'train_safety_network':
            batch = {
                'next_obs': np.stack(self.buffer_next_obs)[batch_indices, :],              # next obs
                'constrain_h': np.stack(self.buffer_constrain_h)[batch_indices],        # constrain
                'done': np.stack(self.buffer_dones)[batch_indices],                     # done
            }
        else:
            raise ValueError(f'Unknown mode {self.mode}')

        return batch

    # def reset_buffer(self):
    #     self.buffer_ego_actions = [[] for _ in range(self.num_scenario)]
    #     self.buffer_scenario_actions = [[] for _ in range(self.num_scenario)]
    #     self.buffer_obs = [[] for _ in range(self.num_scenario)]
    #     self.buffer_next_obs = [[] for _ in range(self.num_scenario)]
    #     self.buffer_rewards = [[] for _ in range(self.num_scenario)]
    #     self.buffer_dones = [[] for _ in range(self.num_scenario)]
    #     self.buffer_additional_dict = [{} for _ in range(self.num_scenario)]
    #     self.buffer_next_additional_dict = [{} for _ in range(self.num_scenario)]

    # def store(self, data_list, additional_dict):
    #     ego_actions = data_list[0]
    #     scenario_actions = data_list[1]
    #     obs = data_list[2]
    #     next_obs = data_list[3]
    #     rewards = data_list[4]
    #     dones = data_list[5]
    #     infos = additional_dict[0]
    #     next_infos = additional_dict[1]
    #     self.buffer_len += len(rewards)
    #     # separate trajectories according to infos
    #     for s_i in range(len(infos)):
    #         sid = infos[s_i]['scenario_id']  # the index of the parallel scenarios
    #         self.buffer_ego_actions[sid].append(ego_actions[s_i])
    #         self.buffer_scenario_actions[sid].append(scenario_actions[s_i])
    #         if self.need_obs:
    #             self.buffer_obs[sid].append(obs[s_i])
    #             self.buffer_next_obs[sid].append(next_obs[s_i])
    #         self.buffer_rewards[sid].append(rewards[s_i])
    #         self.buffer_dones[sid].append(dones[s_i])
    #
    #         # store infos in given dict (e.g. actor_info)
    #         for key in infos[s_i].keys():
    #             if key == 'route_waypoints' or key == 'gps_route' or key == 'route':
    #                 continue
    #             if key not in self.buffer_additional_dict[s_i].keys():
    #                 self.buffer_additional_dict[s_i][key] = []
    #             self.buffer_additional_dict[s_i][key].append(infos[s_i][key])
    #
    #         # store next infos in given dict (e.g. actor_info)
    #         for key in next_infos[s_i].keys():
    #             if key == 'route_waypoints' or key == 'gps_route' or key == 'route':
    #                 continue
    #             if key not in self.buffer_next_additional_dict[s_i].keys():
    #                 self.buffer_next_additional_dict[s_i]['n_' + key] = []
    #             self.buffer_next_additional_dict[s_i]['n_' + key].append(infos[s_i][key])

    # def sample(self, batch_size):
    #     # prepare concatenated list
    #     prepared_ego_actions = []
    #     prepared_scenario_actions = []
    #     prepared_obs = []
    #     prepared_next_obs = []
    #     prepared_rewards = []
    #     prepared_dones = []
    #     prepared_infos = {}
    #     prepared_next_infos = {}
    #
    #     # get the length of each sub-buffer
    #     samples_per_trajectory = self.buffer_capacity // self.num_scenario  # assume average over all sub-buffer
    #     for s_i in range(self.num_scenario):
    #         # select the latest samples starting from the end of buffer
    #         num_trajectory = len(self.buffer_rewards[s_i])
    #         start_idx = np.max([0, num_trajectory - samples_per_trajectory])
    #
    #         # concat
    #         prepared_ego_actions += self.buffer_ego_actions[s_i][start_idx:]
    #         prepared_scenario_actions += self.buffer_scenario_actions[s_i][start_idx:]
    #         if self.need_obs:
    #             prepared_obs += self.buffer_obs[s_i][start_idx:]
    #             prepared_next_obs += self.buffer_next_obs[s_i][start_idx:]
    #         prepared_rewards += self.buffer_rewards[s_i][start_idx:]
    #         prepared_dones += self.buffer_dones[s_i][start_idx:]
    #
    #         # add infos
    #         for k_i in self.buffer_additional_dict[s_i].keys():
    #             if k_i not in prepared_infos.keys():
    #                 prepared_infos[k_i] = []
    #             prepared_infos[k_i] += self.buffer_additional_dict[s_i][k_i][start_idx:]
    #         # add next infos
    #         for k_i in self.buffer_next_additional_dict[s_i].keys():
    #             if k_i not in prepared_next_infos.keys():
    #                 prepared_next_infos[k_i] = []
    #             prepared_next_infos[k_i] += self.buffer_next_additional_dict[s_i][k_i][start_idx:]
    #
    #     # sample from concatenated list
    #     # the first sample does not have previous state ()
    #     sample_index = np.random.randint(1, len(prepared_rewards), size=batch_size)
    #
    #     # prepare batch
    #     action = prepared_ego_actions if self.mode == 'train_agent' else prepared_scenario_actions
    #     batch = {
    #         'action': np.stack(action)[sample_index],                 # action
    #         'reward': np.stack(prepared_rewards)[sample_index],       # reward
    #         'done': np.stack(prepared_dones)[sample_index],           # done
    #     }
    #     if self.need_obs:
    #         batch.update({
    #             'state': np.stack(prepared_obs)[sample_index, :],         # state
    #             'n_state': np.stack(prepared_next_obs)[sample_index, :],  # next state
    #         })
    #
    #     # add additional information to the batch
    #     batch_info = {}
    #     for k_i in prepared_infos.keys():
    #         if k_i == 'route_waypoints' or k_i == 'gps_route' or k_i == 'route':
    #             continue
    #         batch_info[k_i] = np.stack(prepared_infos[k_i])[sample_index]
    #         batch_info['n_' + k_i] = np.stack(prepared_infos[k_i])[sample_index]
    #
    #     # combine two dicts
    #     batch.update(batch_info)
    #     return batch
