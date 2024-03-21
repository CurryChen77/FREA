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
import h5py
import time

import numpy as np
import torch

from safebench.gym_carla.envs.utils import process_ego_action


class ReplayBuffer:
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
            feasibility_config=None,
            buffer_capacity=10000,
            logger=None):
        self.mode = mode
        # define obs shape and action shape for different modes
        if self.mode == 'train_scenario':
            self.obs_shape = tuple(scenario_config['scenario_state_shape'])
            self.action_dim = scenario_config['scenario_action_dim']
            self.state_dim = scenario_config['scenario_state_dim']
        elif self.mode == 'train_agent':
            self.obs_shape = agent_config['ego_state_dim']
            self.action_dim = agent_config['ego_action_dim']
            self.state_dim = agent_config['ego_state_dim']
        else:
            raise ValueError

        self.buffer_capacity = buffer_capacity
        self.num_scenario = num_scenario
        self.agent_need_obs = True if (agent_config['obs_type'] == 'plant') else False
        self.pos = 0
        self.buffer_len = 0
        self.full = False
        self.logger = logger

        if scenario_policy_type == 'offpolicy' and start_episode != 0:
            model_path = os.path.join(scenario_config['ROOT_DIR'], scenario_config['model_path'], scenario_config['CBV_selection'])
            agent_info = 'EgoPolicy_' + scenario_config['agent_policy'] + "-" + scenario_config['agent_obs_type']
            scenario_name = "all" if scenario_config['scenario_id'] is None else 'Scenario' + str(scenario_config['scenario_id'])
            load_dir = os.path.join(model_path, agent_info, scenario_name + "_" + current_map)
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

    def process_CBV_data(self, data_list, additional_dict):
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
            scenario_actions, dones, obs, next_obs, rewards = self.process_CBV_data(data_list, additional_dict)
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

        # get the buffer length
        self.buffer_len = self.buffer_capacity if self.full else self.pos

    def check_scenario_id_for_saving(self, config_len):
        pass

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
        else:
            raise ValueError(f'Unknown mode {self.mode}')

