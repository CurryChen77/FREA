#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：replay_buffer.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""
import os
import pickle
import time

import numpy as np
import torch


class RouteReplayBuffer:
    """
        This buffer supports parallel storing transitions from multiple trajectories.
    """

    def __init__(
            self,
            num_scenario,
            mode,
            start_episode,
            scenario_policy_type=None,
            current_map=None,
            agent_config=None,
            scenario_config=None,
            safety_network_config=None,
            buffer_capacity=10000,
            logger=None):
        self.mode = mode
        # define obs shape and action shape for different modes
        if self.mode == 'train_scenario':
            self.obs_shape = tuple(scenario_config['scenario_state_shape'])
            self.action_dim = scenario_config['scenario_action_dim']
        elif self.mode == 'train_agent':
            self.obs_shape = agent_config['ego_state_dim']
            self.action_dim = agent_config['ego_action_dim']
        elif self.mode == 'train_safety_network':
            self.obs_shape = tuple(safety_network_config['state_shape'])
            self.action_dim = safety_network_config['action_dim']

        self.buffer_capacity = buffer_capacity
        self.num_scenario = num_scenario
        self.agent_need_obs = True if (agent_config['obs_type'] == 'simple_state' or agent_config['obs_type'] == 'plant') else False
        self.safety_network_obs_type = safety_network_config['obs_type'] if safety_network_config else None
        self.pos = 0
        self.buffer_len = 0
        self.full = False
        self.logger = logger

        if scenario_policy_type == 'offpolicy' and start_episode != 0:
            model_path = os.path.join(scenario_config['ROOT_DIR'], scenario_config['model_path'], scenario_config['CBV_selection'])
            agent_info = 'EgoPolicy_' + scenario_config['agent_policy'] + "-" + scenario_config['agent_obs_type']
            safety_network = scenario_config['safety_network']
            scenario_name = "all" if scenario_config['scenario_id'] is None else 'scenario' + str(scenario_config['scenario_id'])
            load_dir = os.path.join(model_path, agent_info, safety_network, scenario_name + "_" + current_map)
            self.load_buffer(dir_path=load_dir, filename=f'buffer.{start_episode:04}.pkl')
        else:
            self.reset_buffer()

    def reset_buffer(self):
        self.pos = 0
        self.buffer_len = 0
        self.full = False
        if self.mode == 'train_scenario':
            self.buffer_actions = np.zeros((self.buffer_capacity, self.action_dim), dtype=np.float32)
            self.buffer_obs = np.zeros((self.buffer_capacity, *self.obs_shape), dtype=np.float32)
            self.buffer_next_obs = np.zeros((self.buffer_capacity, *self.obs_shape), dtype=np.float32)
            self.buffer_rewards = np.zeros(self.buffer_capacity, dtype=np.float32)
            self.buffer_dones = np.zeros(self.buffer_capacity, dtype=np.float32)
        elif self.mode == 'train_agent':
            self.buffer_actions = np.zeros((self.buffer_capacity, self.action_dim), dtype=np.float32)
            self.buffer_obs = np.zeros((self.buffer_capacity, self.obs_shape), dtype=np.float32)
            self.buffer_next_obs = np.zeros((self.buffer_capacity, self.obs_shape), dtype=np.float32)
            self.buffer_rewards = np.zeros(self.buffer_capacity, dtype=np.float32)
            self.buffer_dones = np.zeros(self.buffer_capacity, dtype=np.float32)
        elif self.mode == 'train_safety_network':
            self.buffer_next_obs = np.zeros((self.buffer_capacity, *self.obs_shape), dtype=np.float32)
            self.buffer_constrain_h = np.zeros(self.buffer_capacity, dtype=np.float32)
            self.buffer_dones = np.zeros(self.buffer_capacity, dtype=np.float32)

    def streamline_CBV_data(self, data_list, additional_dict):
        """
            1. remove the meaningless data when CBV is None (scenario obs, next scenario obs, scenario action are all None)
            2. remove the truncated CBV data
            key: the done came from the CBV view instead of ego view
        """
        assert len(additional_dict[0]) == len(additional_dict[1]), "the length of info and next_info should be the same"
        scenario_actions = []
        terminated = []
        obs = []
        next_obs = []
        rewards = []
        for actions, infos, next_infos in zip(data_list[1], additional_dict[0], additional_dict[1]):
            assert len(actions) == len(next_infos['CBVs_obs']) == len(infos['CBVs_obs']) \
                == len(next_infos['CBVs_reward']) == len(next_infos['CBVs_truncated']), "length of the trajectory should be the same"
            for CBV_id in actions.keys():
                if next_infos['CBVs_truncated'][CBV_id] is not True:
                    # store the trajectory that is not truncated
                    scenario_actions.append(actions[CBV_id])
                    terminated.append(next_infos['CBVs_terminated'][CBV_id])
                    obs.append(infos['CBVs_obs'][CBV_id])
                    next_obs.append(next_infos['CBVs_obs'][CBV_id])
                    rewards.append(next_infos['CBVs_reward'][CBV_id])
        return scenario_actions, terminated, obs, next_obs, rewards

    def store(self, data_list, additional_dict):
        """
            additional dict[0]: the current infos
            additional dict[1]: the next infos
        """
        # store for scenario training
        if self.mode == 'train_scenario':
            scenario_actions, dones, obs, next_obs, rewards = self.streamline_CBV_data(data_list, additional_dict)
            for s_i in range(len(dones)):
                self.buffer_actions[self.pos] = np.array(scenario_actions[s_i])
                self.buffer_obs[self.pos] = np.array(obs[s_i])  # CBV_obs
                self.buffer_next_obs[self.pos] = np.array(next_obs[s_i])  # CBV next obs from next info
                self.buffer_rewards[self.pos] = np.array(rewards[s_i])  # CBV reward
                self.buffer_dones[self.pos] = np.array(dones[s_i])
                self.pos += 1
                if self.pos == self.buffer_capacity:
                    self.full = True
                    self.pos = 0

        # store for agent training
        elif self.mode == 'train_agent':
            ego_actions = data_list[0]
            obs = data_list[2]
            next_obs = data_list[3]
            rewards = data_list[4]
            dones = data_list[5]
            for s_i in range(len(dones)):
                self.buffer_actions[self.pos] = np.array(ego_actions[s_i])
                self.buffer_obs[self.pos] = np.array(obs[s_i])
                self.buffer_next_obs[self.pos] = np.array(next_obs[s_i])
                self.buffer_rewards[self.pos] = np.array(rewards[s_i])
                self.buffer_dones[self.pos] = np.array(dones[s_i])
                self.pos += 1
                self.buffer_len += 1
                if self.pos == self.buffer_capacity:
                    self.full = True
                    self.pos = 0

        # store for agent training
        elif self.mode == 'train_safety_network':
            dones = data_list[5]
            if self.safety_network_obs_type == 'ego_info':
                next_obs = [info['ego_info'] for info in additional_dict[1]]  # additional_dict[1] means the next info
            else:
                raise ValueError(f'Unknown safety_network obs_type')
            constrain_h = [info['constrain_h'] for info in additional_dict[0]]  # the constraint h should from infos

            for s_i in range(len(dones)):
                self.buffer_next_obs[self.pos] = np.array(next_obs[s_i])
                self.buffer_constrain_h[self.pos] = np.array(constrain_h[s_i])
                self.buffer_dones[self.pos] = np.array(dones[s_i])
                self.pos += 1
                self.buffer_len += 1
                if self.pos == self.buffer_capacity:
                    self.full = True
                    self.pos = 0

        # get the buffer length
        self.buffer_len = self.buffer_capacity if self.full else self.pos

    def sample(self, batch_size):
        upper_bound = self.buffer_capacity if self.full else self.pos
        batch_indices = np.random.randint(0, high=upper_bound, size=batch_size)

        if self.mode == 'train_scenario':
            batch = {
                'action': np.stack(self.buffer_actions)[batch_indices],  # action
                'CBVs_obs': np.stack(self.buffer_obs)[batch_indices, :],  # obs
                'next_CBVs_obs': np.stack(self.buffer_next_obs)[batch_indices, :],  # next obs
                'reward': np.stack(self.buffer_rewards)[batch_indices],  # reward
                'done': np.stack(self.buffer_dones)[batch_indices],  # done
            }
        elif self.mode == 'train_agent':
            batch = {
                'action': np.stack(self.buffer_actions)[batch_indices],  # action
                'obs': np.stack(self.buffer_obs)[batch_indices, :],  # obs
                'next_obs': np.stack(self.buffer_next_obs)[batch_indices, :],  # next obs
                'reward': np.stack(self.buffer_rewards)[batch_indices],  # reward
                'done': np.stack(self.buffer_dones)[batch_indices],  # done
            }
        elif self.mode == 'train_safety_network':
            batch = {
                'next_obs': np.stack(self.buffer_next_obs)[batch_indices, :],  # next obs
                'constrain_h': np.stack(self.buffer_constrain_h)[batch_indices],  # constrain
                'done': np.stack(self.buffer_dones)[batch_indices],  # done
            }
        else:
            raise ValueError(f'Unknown mode {self.mode}')

        return batch

    def save_buffer(self, dir_path, filename):
        if self.mode == 'train_scenario':
            buffer = {
                'buffer_actions': self.buffer_actions,
                'buffer_obs': self.buffer_obs,
                'buffer_next_obs': self.buffer_next_obs,
                'buffer_rewards': self.buffer_rewards,
                'buffer_dones': self.buffer_dones,
                'buffer_infos': [self.pos, self.buffer_len, self.full]
            }
        elif self.mode == 'train_agent':
            buffer = {
                'buffer_actions': self.buffer_actions,
                'buffer_obs': self.buffer_obs,
                'buffer_next_obs': self.buffer_next_obs,
                'buffer_rewards': self.buffer_rewards,
                'buffer_dones': self.buffer_dones,
                'buffer_infos': [self.pos, self.buffer_len, self.full]
            }
        elif self.mode == 'train_safety_network':
            buffer = {
                'buffer_next_obs': self.buffer_next_obs,
                'buffer_constrain_h': self.buffer_constrain_h,
                'buffer_dones': self.buffer_dones,
                'buffer_infos': [self.pos, self.buffer_len, self.full]
            }
        else:
            raise ValueError(f'Unknown mode {self.mode}')

        path = os.path.join(dir_path, filename)
        with open(path, 'wb') as file:
            pickle.dump(buffer, file)

    def load_buffer(self, dir_path, filename):
        path = os.path.join(dir_path, filename)
        if os.path.exists(path):
            try:
                with open(path, 'rb') as file:
                    buffer = pickle.load(file)
                    self.logger.log(f'>> Successfully loading the replay buffer checkpoint', 'yellow')
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            print(f"File {filename} not found in {dir_path}.")

        if self.mode == 'train_scenario':
            self.buffer_actions = buffer['buffer_actions']
            self.buffer_obs = buffer['buffer_obs']
            self.buffer_next_obs = buffer['buffer_next_obs']
            self.buffer_rewards = buffer['buffer_rewards']
            self.buffer_dones = buffer['buffer_dones']
            self.pos = buffer['buffer_infos'][0]
            self.buffer_len = buffer['buffer_infos'][1]
            self.full = buffer['buffer_infos'][2]
        elif self.mode == 'train_agent':
            self.buffer_actions = buffer['buffer_actions']
            self.buffer_obs = buffer['buffer_obs']
            self.buffer_next_obs = buffer['buffer_next_obs']
            self.buffer_rewards = buffer['buffer_rewards']
            self.buffer_dones = buffer['buffer_dones']
            self.pos = buffer['buffer_infos'][0]
            self.buffer_len = buffer['buffer_infos'][1]
            self.full = buffer['buffer_infos'][2]
        elif self.mode == 'train_safety_network':
            self.buffer_next_obs = buffer['buffer_next_obs']
            self.buffer_constrain_h = buffer['buffer_constrain_h']
            self.buffer_dones = buffer['buffer_dones']
            self.pos = buffer['buffer_infos'][0]
            self.buffer_len = buffer['buffer_infos'][1]
            self.full = buffer['buffer_infos'][2]
        else:
            raise ValueError(f'Unknown mode {self.mode}')

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
