#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：buffer.py
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
            safety_network_config=None,
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
        elif self.mode == 'train_safety_network':
            self.obs_shape = tuple(safety_network_config['state_shape'])
            self.action_dim = safety_network_config['action_dim']
            self.state_dim = safety_network_config['state_dim']
        else:
            raise ValueError

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


class RolloutBuffer:
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
        self.temp_buffer = {'actions': {}, 'obs': {}, 'next_obs': {}, 'rewards': {}, 'dones': {}}
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
        elif self.mode == 'train_safety_network':
            self.obs_shape = tuple(safety_network_config['state_shape'])
            self.action_dim = safety_network_config['action_dim']
            self.state_dim = safety_network_config['state_dim']
        else:
            raise ValueError

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
        self.temp_buffer = {'actions': {}, 'log_probs': {}, 'obs': {}, 'next_obs': {}, 'rewards': {}, 'dones': {}}
        if self.mode == 'train_scenario':
            self.buffer_actions = np.zeros((self.buffer_capacity, self.action_dim), dtype=np.float32)
            self.buffer_log_probs = np.zeros((self.buffer_capacity), dtype=np.float32)
            self.buffer_obs = np.zeros((self.buffer_capacity, *self.obs_shape), dtype=np.float32)
            self.buffer_next_obs = np.zeros((self.buffer_capacity, *self.obs_shape), dtype=np.float32)
            self.buffer_rewards = np.zeros(self.buffer_capacity, dtype=np.float32)
            self.buffer_dones = np.zeros(self.buffer_capacity, dtype=np.float32)
        elif self.mode == 'train_agent':
            self.buffer_actions = np.zeros((self.buffer_capacity, self.action_dim), dtype=np.float32)
            self.buffer_log_probs = np.zeros((self.buffer_capacity), dtype=np.float32)
            self.buffer_obs = np.zeros((self.buffer_capacity, self.obs_shape), dtype=np.float32)
            self.buffer_next_obs = np.zeros((self.buffer_capacity, self.obs_shape), dtype=np.float32)
            self.buffer_rewards = np.zeros(self.buffer_capacity, dtype=np.float32)
            self.buffer_dones = np.zeros(self.buffer_capacity, dtype=np.float32)
        elif self.mode == 'train_safety_network':
            self.buffer_next_obs = np.zeros((self.buffer_capacity, *self.obs_shape), dtype=np.float32)
            self.buffer_constrain_h = np.zeros(self.buffer_capacity, dtype=np.float32)
            self.buffer_dones = np.zeros(self.buffer_capacity, dtype=np.float32)

    def process_CBV_data(self, data_list, additional_dict):
        """
            1. remove the meaningless data when CBV is None (scenario obs, next scenario obs, scenario action are all None)
            2. remove the truncated CBV data
            key: the done came from the CBV view instead of ego view
        """
        processed_actions, processed_log_probs, processed_obs, processed_next_obs, processed_rewards, processed_dones = [], [], [], [], [], []
        scenario_actions, scenario_log_probs = data_list[1]
        assert len(scenario_actions) == len(additional_dict[0]) == len(additional_dict[1]), "the length of info and next_info should be the same"
        for actions, log_probs, infos, next_infos in zip(scenario_actions, scenario_log_probs, additional_dict[0], additional_dict[1]):
            assert len(actions) == len(next_infos['CBVs_obs']) == len(infos['CBVs_obs']) \
                == len(next_infos['CBVs_reward']) == len(next_infos['CBVs_truncated']), "length of the trajectory should be the same"
            for CBV_id in actions.keys():
                # if the first CBV in the history, create an empty list for all the trajectory
                if CBV_id not in self.temp_buffer['log_probs'].keys():
                    self.temp_buffer['actions'][CBV_id] = []
                    self.temp_buffer['log_probs'][CBV_id] = []
                    self.temp_buffer['obs'][CBV_id] = []
                    self.temp_buffer['next_obs'][CBV_id] = []
                    self.temp_buffer['rewards'][CBV_id] = []
                    self.temp_buffer['dones'][CBV_id] = []

                # add one-step trajectory in to the corresponding CBV dict
                self.temp_buffer['actions'][CBV_id].append(actions[CBV_id])

                self.temp_buffer['log_probs'][CBV_id].append(log_probs[CBV_id])
                self.temp_buffer['obs'][CBV_id].append(infos['CBVs_obs'][CBV_id])
                self.temp_buffer['next_obs'][CBV_id].append(next_infos['CBVs_obs'][CBV_id])
                self.temp_buffer['rewards'][CBV_id].append(next_infos['CBVs_reward'][CBV_id])
                if next_infos['CBVs_terminated'][CBV_id] or next_infos['CBVs_truncated'][CBV_id]:
                    self.temp_buffer['dones'][CBV_id].append(True)
                    # the continuous trajectory of the CBV is completed, copy the whole trajectory into the list for further storing
                    processed_actions.extend(self.temp_buffer['actions'].pop(CBV_id))
                    processed_log_probs.extend(self.temp_buffer['log_probs'].pop(CBV_id))
                    processed_obs.extend(self.temp_buffer['obs'].pop(CBV_id))
                    processed_next_obs.extend(self.temp_buffer['next_obs'].pop(CBV_id))
                    processed_rewards.extend(self.temp_buffer['rewards'].pop(CBV_id))
                    processed_dones.extend(self.temp_buffer['dones'].pop(CBV_id))
                else:
                    self.temp_buffer['dones'][CBV_id].append(False)

        return processed_actions, processed_log_probs, processed_obs, processed_next_obs, processed_rewards, processed_dones

    def store(self, data_list, additional_dict):
        """
            additional dict[0]: the current infos
            additional dict[1]: the next infos
        """
        # store for scenario training
        if self.mode == 'train_scenario':
            scenario_actions, scenario_log_probs, obs, next_obs, rewards, dones = self.process_CBV_data(data_list, additional_dict)
            length = len(dones)
            if length > 0:
                if self.pos + length >= self.buffer_capacity:
                    self.full = True
                    self.pos = 0

                self.buffer_actions[self.pos:self.pos+length, :] = np.array(scenario_actions)
                self.buffer_log_probs[self.pos:self.pos+length] = np.array(scenario_log_probs)
                self.buffer_obs[self.pos:self.pos+length, :] = np.array(obs)  # CBV_obs
                self.buffer_next_obs[self.pos:self.pos+length, :] = np.array(next_obs)  # CBV next obs from next info
                self.buffer_rewards[self.pos:self.pos+length] = np.array(rewards)  # CBV reward
                self.buffer_dones[self.pos:self.pos+length] = np.array(dones)
                self.pos += length

        # store for agent training
        elif self.mode == 'train_agent':
            ego_actions, ego_log_probs = data_list[0]
            obs = data_list[2]
            next_obs = data_list[3]
            rewards = data_list[4]
            dones = data_list[5]
            length = len(dones)
            if length > 0:
                if self.pos + length >= self.buffer_capacity:
                    self.full = True
                    self.pos = 0

            self.buffer_actions[self.pos:self.pos + length, :] = np.array(ego_actions)
            self.buffer_log_probs[self.pos:self.pos + length] = np.array(ego_log_probs)
            self.buffer_obs[self.pos:self.pos + length, :] = np.array(obs)  # CBV_obs
            self.buffer_next_obs[self.pos:self.pos + length, :] = np.array(next_obs)  # CBV next obs from next info
            self.buffer_rewards[self.pos:self.pos + length] = np.array(rewards)  # CBV reward
            self.buffer_dones[self.pos:self.pos + length] = np.array(dones)
            self.pos += length

        # TODO store for agent training
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

    def get(self):
        upper_bound = self.buffer_capacity if self.full else self.pos
        if self.mode == 'train_scenario':
            batch = {
                'actions': self.buffer_actions[:upper_bound, ...],  # action
                'log_probs': self.buffer_log_probs[:upper_bound],  # action log probability
                'obs': self.buffer_obs[:upper_bound, ...].reshape((-1, self.state_dim)),  # obs
                'next_obs': self.buffer_next_obs[:upper_bound, ...].reshape((-1, self.state_dim)),  # next obs
                'rewards': self.buffer_rewards[:upper_bound],  # reward
                'dones': self.buffer_dones[:upper_bound],  # done
            }
        elif self.mode == 'train_agent':
            batch = {
                'actions': self.buffer_actions[:upper_bound, ...],  # action
                'log_probs': self.buffer_log_probs[:upper_bound],  # action log probability
                'obs': self.buffer_obs[:upper_bound, ...].reshape((-1, self.state_dim)),  # obs
                'next_obs': self.buffer_next_obs[:upper_bound, ...].reshape((-1, self.state_dim)),  # next obs
                'rewards': self.buffer_rewards[:upper_bound],  # reward
                'dones': self.buffer_dones[:upper_bound],  # done
            }
        elif self.mode == 'train_safety_network':
            # TODO
            batch = {
                'next_obs': self.buffer_next_obs[:upper_bound, :],  # next obs
                'constrain_h': self.buffer_constrain_h[:upper_bound],  # constrain
                'done': self.buffer_dones[:upper_bound],  # done
            }
        else:
            raise ValueError(f'Unknown mode {self.mode}')

        return batch