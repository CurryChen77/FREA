#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：rollout_buffer.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""
import h5py
import numpy as np

from frea.gym_carla.envs.utils import process_ego_action


class RolloutBuffer:
    """
        This buffer supports parallel storing transitions from multiple trajectories.
    """

    def __init__(
            self,
            num_scenario,
            mode,
            agent_config=None,
            scenario_config=None,
            feasibility_config=None,
            buffer_capacity=10000,
            logger=None):
        self.mode = mode
        self.buffer_capacity = buffer_capacity
        self.num_scenario = num_scenario
        self.buffer_len = 0
        self.store_scenario_id = None  # if scenario config length < self.num_scenario happens, this value will no longer be None
        # define obs shape and action shape for different modes
        if self.mode == 'train_scenario':
            self.temp_buffer = {'actions': {}, 'obs': {}, 'next_obs': {}, 'rewards': {}, 'dones': {}}
            self.obs_shape = tuple(scenario_config['scenario_state_shape'])
            self.action_dim = scenario_config['scenario_action_dim']
            self.state_dim = scenario_config['scenario_state_dim']
            self.ego_learnable = True if agent_config['learnable'] else False
            self.ego_onpolicy = True if agent_config['learnable'] and agent_config['onpolicy'] else False
            # if the scenario agent needs the feasibility
            self.use_feasibility = scenario_config['feasibility']
            if self.use_feasibility:
                self.feasibility_obs_shape = tuple(feasibility_config['state_shape'])
                self.feasibility_action_dim = feasibility_config['agent_action_dim']
            self.scenario_pos = 0
            self.scenario_full = False
        elif self.mode == 'train_agent':
            self.obs_shape = tuple([agent_config['ego_state_dim']])
            self.action_dim = agent_config['ego_action_dim']
            self.state_dim = agent_config['ego_state_dim']
            self.agent_pos = [0] * self.num_scenario
            self.agent_full = [False] * self.num_scenario
        elif self.mode == 'collect_feasibility_data':
            self.obs_shape = tuple(feasibility_config['state_shape'])
            self.action_dim = feasibility_config['agent_action_dim']  # the action dim of the safety network is the ego action
            self.state_dim = feasibility_config['feasibility_state_dim']
            self.feasibility_pos = [0] * self.num_scenario
            self.feasibility_full = [False] * self.num_scenario
            self.feasibility_ego_onpolicy = True if agent_config['learnable'] and agent_config['onpolicy'] else False
            self.feasibility_search_radius = feasibility_config['search_radius']
            self.feasibility_acc_range = feasibility_config['acc_range']
            self.feasibility_steer_range = feasibility_config['steer_range']
            self.feasibility_ego_agent_learnable = feasibility_config['ego_agent_learnable']
        else:
            raise ValueError

        self.feasibility_obs_type = feasibility_config['obs_type'] if feasibility_config else None

        self.logger = logger

        self.reset_buffer()

    def reset_buffer(self):
        self.buffer_len = 0
        if self.mode == 'train_scenario':
            self.scenario_pos = 0
            self.scenario_full = False
            self.temp_buffer = {'actions': {}, 'log_probs': {}, 'obs': {}, 'next_obs': {}, 'rewards': {}, 'dones': {}, 'terminated': {}}
            self.buffer_actions = np.zeros((self.buffer_capacity, self.action_dim), dtype=np.float32)
            self.buffer_log_probs = np.zeros(self.buffer_capacity, dtype=np.float32)
            self.buffer_obs = np.zeros((self.buffer_capacity, *self.obs_shape), dtype=np.float32)
            self.buffer_next_obs = np.zeros((self.buffer_capacity, *self.obs_shape), dtype=np.float32)
            self.buffer_rewards = np.zeros(self.buffer_capacity, dtype=np.float32)
            self.buffer_dones = np.zeros(self.buffer_capacity, dtype=np.float32)
            self.buffer_terminated = np.zeros((self.buffer_capacity), dtype=np.float32)
            if self.use_feasibility:
                self.buffer_ego_actions = np.zeros((self.buffer_capacity, self.feasibility_action_dim), dtype=np.float32)
                self.buffer_ego_CBV_dis = np.zeros(self.buffer_capacity, dtype=np.float32)
                self.buffer_ego_CBV_next_dis = np.zeros(self.buffer_capacity, dtype=np.float32)
                self.buffer_ego_CBV_obs = np.zeros((self.buffer_capacity, *self.feasibility_obs_shape), dtype=np.float32)
                self.buffer_ego_CBV_next_obs = np.zeros((self.buffer_capacity, *self.feasibility_obs_shape), dtype=np.float32)
                self.temp_buffer.update({
                    'ego_actions': {}, 'ego_CBV_obs': {}, 'ego_CBV_next_obs': {},
                    'ego_CBV_dis': {}, 'ego_CBV_next_dis': {}})
        elif self.mode == 'train_agent':
            self.agent_pos = [0] * self.num_scenario
            self.agent_full = [False] * self.num_scenario
            self.buffer_actions = np.zeros((self.buffer_capacity, self.num_scenario, self.action_dim), dtype=np.float32)
            self.buffer_log_probs = np.zeros((self.buffer_capacity, self.num_scenario), dtype=np.float32)
            self.buffer_obs = np.zeros((self.buffer_capacity, self.num_scenario, *self.obs_shape), dtype=np.float32)
            self.buffer_next_obs = np.zeros((self.buffer_capacity, self.num_scenario, *self.obs_shape), dtype=np.float32)
            self.buffer_rewards = np.zeros((self.buffer_capacity, self.num_scenario), dtype=np.float32)
            self.buffer_dones = np.zeros((self.buffer_capacity, self.num_scenario), dtype=np.float32)
        elif self.mode == 'collect_feasibility_data':
            self.feasibility_pos = [0] * self.num_scenario
            self.feasibility_full = [False] * self.num_scenario
            self.buffer_actions = np.zeros((self.buffer_capacity, self.num_scenario, self.action_dim), dtype=np.float32)
            self.buffer_obs = np.zeros((self.buffer_capacity, self.num_scenario, *self.obs_shape), dtype=np.float32)
            self.buffer_next_obs = np.zeros((self.buffer_capacity, self.num_scenario, *self.obs_shape), dtype=np.float32)
            self.buffer_ego_min_dis = np.zeros((self.buffer_capacity, self.num_scenario), dtype=np.float32)
            self.buffer_ego_collide = np.zeros((self.buffer_capacity, self.num_scenario), dtype=np.float32)
            self.buffer_dones = np.zeros((self.buffer_capacity, self.num_scenario), dtype=np.float32)

    def process_CBV_data(self, data_list, additional_dict):
        """
            1. remove the meaningless data when CBV is None (scenario obs, next scenario obs, scenario action are all None)
            key: the done came from the CBV view instead of ego view
        """
        # the processed data for normal scenario training
        processed_actions, processed_log_probs, processed_obs, processed_next_obs, processed_rewards, processed_dones, processed_terminated = [], [], [], [], [], [], []
        # the processed data for feasibility
        processed_ego_actions, processed_ego_CBV_obs, processed_ego_CBV_next_obs = [], [], []
        processed_ego_CBV_dis, processed_ego_CBV_next_dis = [], []
        # process the ego actions
        if self.ego_onpolicy:
            all_ego_actions, _ = data_list[0]  # ego actions are in datalist[0]
        else:
            all_ego_actions = data_list[0]

        # get the scenario actions
        all_scenario_actions, all_scenario_log_probs = data_list[1]

        assert len(all_scenario_actions) == len(additional_dict[0]) == len(additional_dict[1]), "the length of info and next_info should be the same"
        # Traverse all the step data in all scenarios
        for ego_action, actions, log_probs, infos, next_infos in zip(all_ego_actions, all_scenario_actions, all_scenario_log_probs, additional_dict[0], additional_dict[1]):
            assert len(actions) == len(next_infos['CBVs_obs']) == len(infos['CBVs_obs']) \
                   == len(next_infos['CBVs_reward']) == len(next_infos['CBVs_truncated']), "length of the trajectory should be the same"
            # if learnable ego agent, need to process the action
            if self.use_feasibility and self.ego_learnable:
                ego_action = process_ego_action(ego_action, acc_range=[-3.0, 3.0], steering_range=[-0.3, 0.3])

            # Traverse all the CBVs in one scenario
            for CBV_id in actions.keys():
                # if the first CBV in the history, create an empty list for all the trajectory
                if CBV_id not in self.temp_buffer['actions'].keys():
                    self.temp_buffer['actions'][CBV_id] = []
                    self.temp_buffer['log_probs'][CBV_id] = []
                    self.temp_buffer['obs'][CBV_id] = []
                    self.temp_buffer['next_obs'][CBV_id] = []
                    self.temp_buffer['rewards'][CBV_id] = []
                    self.temp_buffer['dones'][CBV_id] = []
                    self.temp_buffer['terminated'][CBV_id] = []
                    if self.use_feasibility:
                        self.temp_buffer['ego_CBV_obs'][CBV_id] = []
                        self.temp_buffer['ego_CBV_next_obs'][CBV_id] = []
                        self.temp_buffer['ego_actions'][CBV_id] = []
                        self.temp_buffer['ego_CBV_dis'][CBV_id] = []
                        self.temp_buffer['ego_CBV_next_dis'][CBV_id] = []

                # add one-step trajectory in to the corresponding CBV dict
                self.temp_buffer['actions'][CBV_id].append(actions[CBV_id])
                self.temp_buffer['log_probs'][CBV_id].append(log_probs[CBV_id])
                self.temp_buffer['obs'][CBV_id].append(infos['CBVs_obs'][CBV_id])
                self.temp_buffer['next_obs'][CBV_id].append(next_infos['CBVs_obs'][CBV_id])
                self.temp_buffer['rewards'][CBV_id].append(next_infos['CBVs_reward'][CBV_id])
                if self.use_feasibility:
                    self.temp_buffer['ego_CBV_obs'][CBV_id].append(infos['ego_CBV_obs'][CBV_id])
                    self.temp_buffer['ego_actions'][CBV_id].append(ego_action)
                    self.temp_buffer['ego_CBV_next_obs'][CBV_id].append(next_infos['ego_CBV_obs'][CBV_id])
                    self.temp_buffer['ego_CBV_dis'][CBV_id].append(infos['ego_CBV_dis'][CBV_id])
                    self.temp_buffer['ego_CBV_next_dis'][CBV_id].append(next_infos['ego_CBV_dis'][CBV_id])

                if next_infos['CBVs_terminated'][CBV_id] or next_infos['CBVs_truncated'][CBV_id]:
                    self.temp_buffer['dones'][CBV_id].append(True)
                    self.temp_buffer['terminated'][CBV_id].append(next_infos['CBVs_terminated'][CBV_id])
                    # the continuous trajectory of the CBV is completed, copy the whole trajectory into the list for further storing
                    processed_actions.extend(self.temp_buffer['actions'].pop(CBV_id))
                    processed_log_probs.extend(self.temp_buffer['log_probs'].pop(CBV_id))
                    processed_obs.extend(self.temp_buffer['obs'].pop(CBV_id))
                    processed_next_obs.extend(self.temp_buffer['next_obs'].pop(CBV_id))
                    processed_rewards.extend(self.temp_buffer['rewards'].pop(CBV_id))
                    processed_dones.extend(self.temp_buffer['dones'].pop(CBV_id))
                    processed_terminated.extend(self.temp_buffer['terminated'].pop(CBV_id))
                    if self.use_feasibility:
                        processed_ego_actions.extend(self.temp_buffer['ego_actions'].pop(CBV_id))
                        processed_ego_CBV_obs.extend(self.temp_buffer['ego_CBV_obs'].pop(CBV_id))
                        processed_ego_CBV_next_obs.extend(self.temp_buffer['ego_CBV_next_obs'].pop(CBV_id))
                        processed_ego_CBV_dis.extend(self.temp_buffer['ego_CBV_dis'].pop(CBV_id))
                        processed_ego_CBV_next_dis.extend(self.temp_buffer['ego_CBV_next_dis'].pop(CBV_id))
                else:
                    self.temp_buffer['dones'][CBV_id].append(False)
                    self.temp_buffer['terminated'][CBV_id].append(False)

        return processed_actions, processed_log_probs, processed_obs, processed_next_obs, processed_rewards, processed_dones, processed_terminated, \
            processed_ego_actions, processed_ego_CBV_obs, processed_ego_CBV_next_obs, processed_ego_CBV_dis, processed_ego_CBV_next_dis

    def store(self, data_list, additional_dict):
        """
            additional dict[0]: the current infos
            additional dict[1]: the next infos
        """
        # store for scenario training
        if self.mode == 'train_scenario':
            (scenario_actions, scenario_log_probs, obs, next_obs, rewards, dones, terminated, ego_actions, ego_CBV_obs, ego_CBV_next_obs,
             ego_CBV_dis, ego_CBV_next_dis) = self.process_CBV_data(data_list, additional_dict)

            length = len(dones)
            if length > 10:  # remove the too short CBV trajectory
                if self.scenario_pos + length >= self.buffer_capacity:
                    # the buffer can just hold part of the trajectory dta
                    for i in range(length):
                        if self.scenario_pos < self.buffer_capacity:
                            self.buffer_actions[self.scenario_pos, :] = np.array(scenario_actions[i])
                            self.buffer_log_probs[self.scenario_pos] = np.array(scenario_log_probs[i])
                            self.buffer_obs[self.scenario_pos, :] = np.array(obs[i])  # CBV_obs
                            self.buffer_next_obs[self.scenario_pos, :] = np.array(next_obs[i])  # CBV next obs from next info
                            self.buffer_rewards[self.scenario_pos] = np.array(rewards[i])  # CBV reward
                            self.buffer_dones[self.scenario_pos] = np.array(dones[i])
                            self.buffer_terminated[self.scenario_pos] = np.array(terminated[i])
                            if self.use_feasibility:
                                self.buffer_ego_actions[self.scenario_pos] = np.array(ego_actions[i])
                                self.buffer_ego_CBV_obs[self.scenario_pos] = np.array(ego_CBV_obs[i])
                                self.buffer_ego_CBV_next_obs[self.scenario_pos] = np.array(ego_CBV_next_obs[i])
                                self.buffer_ego_CBV_dis[self.scenario_pos] = np.array(ego_CBV_dis[i])
                                self.buffer_ego_CBV_next_dis[self.scenario_pos] = np.array(ego_CBV_next_dis[i])
                            self.scenario_pos += 1
                        else:
                            break
                    self.scenario_full = True
                else:
                    # the buffer still can hold the whole trajectory
                    self.buffer_actions[self.scenario_pos:self.scenario_pos + length, :] = np.array(scenario_actions)
                    self.buffer_log_probs[self.scenario_pos:self.scenario_pos + length] = np.array(scenario_log_probs)
                    self.buffer_obs[self.scenario_pos:self.scenario_pos + length, :] = np.array(obs)  # CBV_obs
                    self.buffer_next_obs[self.scenario_pos:self.scenario_pos + length, :] = np.array(next_obs)  # CBV next obs from next info
                    self.buffer_rewards[self.scenario_pos:self.scenario_pos + length] = np.array(rewards)  # CBV reward
                    self.buffer_dones[self.scenario_pos:self.scenario_pos + length] = np.array(dones)
                    self.buffer_terminated[self.scenario_pos:self.scenario_pos + length] = np.array(terminated)
                    if self.use_feasibility:
                        self.buffer_ego_CBV_obs[self.scenario_pos:self.scenario_pos + length] = np.array(ego_CBV_obs)
                        self.buffer_ego_CBV_next_obs[self.scenario_pos:self.scenario_pos + length] = np.array(ego_CBV_next_obs)
                        self.buffer_ego_actions[self.scenario_pos:self.scenario_pos + length] = np.array(ego_actions)
                        self.buffer_ego_CBV_dis[self.scenario_pos:self.scenario_pos + length] = np.array(ego_CBV_dis)
                        self.buffer_ego_CBV_next_dis[self.scenario_pos:self.scenario_pos + length] = np.array(ego_CBV_next_dis)
                    self.scenario_pos += length

            # get the buffer length
            self.buffer_len = self.buffer_capacity if self.scenario_full else self.scenario_pos

        # store for agent training
        elif self.mode == 'train_agent':
            all_agent_actions, all_agent_log_probs = data_list[0]  # agent actions are in data_list[0]
            all_obs = data_list[2]
            all_next_obs = data_list[3]
            all_rewards = data_list[4]
            all_dones = data_list[5]
            all_next_infos = additional_dict[1]

            assert len(all_agent_actions) == len(all_obs) == len(all_next_obs) == len(all_rewards) == len(all_dones), "the length of trajectory should be the same"
            for ego_actions, ego_log_probs, obs, next_obs, rewards, dones, next_infos in zip(all_agent_actions, all_agent_log_probs, all_obs, all_next_obs, all_rewards, all_dones,
                                                                                             all_next_infos):
                scenario_id = next_infos['scenario_id'] if self.store_scenario_id is None else self.store_scenario_id
                if not self.agent_full[scenario_id]:
                    self.buffer_actions[self.agent_pos[scenario_id], scenario_id, :] = np.array(ego_actions)
                    self.buffer_log_probs[self.agent_pos[scenario_id], scenario_id] = np.array(ego_log_probs)
                    self.buffer_obs[self.agent_pos[scenario_id], scenario_id, :] = np.array(obs)  # CBV_obs
                    self.buffer_next_obs[self.agent_pos[scenario_id], scenario_id, :] = np.array(next_obs)  # CBV next obs from next info
                    self.buffer_rewards[self.agent_pos[scenario_id], scenario_id] = np.array(rewards)  # CBV reward
                    self.buffer_dones[self.agent_pos[scenario_id], scenario_id] = np.array(dones)
                    self.agent_pos[scenario_id] += 1
                    if self.agent_pos[scenario_id] > self.buffer_capacity // 2:
                        self.agent_full[scenario_id] = True

            buffer_len = 0
            # get the buffer length
            for pos, full in zip(self.agent_pos, self.agent_full):
                buffer_len += self.buffer_capacity // 2 if full else pos
            self.buffer_len = buffer_len

        elif self.mode == 'collect_feasibility_data':
            if self.feasibility_ego_onpolicy:
                # onpolicy ego agent got log prob
                all_actions, _ = data_list[0]
            else:
                all_actions = data_list[0]
            all_dones = data_list[5]
            all_infos = additional_dict[0]
            all_next_infos = additional_dict[1]

            for action, done, infos, next_infos in zip(all_actions, all_dones, all_infos, all_next_infos):
                # if scenario config length < number_scenarios, need to store data on "self.store_scenario_id" index
                scenario_id = next_infos['scenario_id'] if self.store_scenario_id is None else self.store_scenario_id
                ego_min_dis = next_infos['ego_min_dis']  # the ego_min_dis come from the next state
                ego_collide = next_infos['ego_collide']
                obs = infos[self.feasibility_obs_type]
                next_obs = next_infos[self.feasibility_obs_type]

                if self.feasibility_ego_agent_learnable:
                    # if the ego agent is a learnable method, need a further process to convert the original action to throttle, steer, brake of ego
                    action = process_ego_action(action, self.feasibility_acc_range, self.feasibility_steer_range)

                if not self.feasibility_full[scenario_id] and ego_min_dis < self.feasibility_search_radius:
                    self.buffer_actions[self.feasibility_pos[scenario_id], scenario_id, :] = np.array(action)
                    self.buffer_obs[self.feasibility_pos[scenario_id], scenario_id, :] = np.array(obs)
                    self.buffer_next_obs[self.feasibility_pos[scenario_id], scenario_id, :] = np.array(next_obs)
                    self.buffer_dones[self.feasibility_pos[scenario_id], scenario_id] = np.array(done)
                    self.buffer_ego_min_dis[self.feasibility_pos[scenario_id], scenario_id] = np.array(ego_min_dis)
                    self.buffer_ego_collide[self.feasibility_pos[scenario_id], scenario_id] = np.array(ego_collide)
                    self.feasibility_pos[scenario_id] += 1
                    if self.feasibility_pos[scenario_id] > self.buffer_capacity // 2:
                        self.feasibility_full[scenario_id] = True

            buffer_len = 0
            # get the buffer length
            for pos, full in zip(self.feasibility_pos, self.feasibility_full):
                buffer_len += self.buffer_capacity // 2 if full else pos
            self.buffer_len = buffer_len

    def get(self):
        if self.mode == 'train_scenario':
            upper_bound = self.buffer_capacity if self.scenario_full else self.scenario_pos
            batch = {
                'actions': self.buffer_actions[:upper_bound, ...],  # action
                'log_probs': self.buffer_log_probs[:upper_bound],  # action log probability
                'obs': self.buffer_obs[:upper_bound, ...].reshape((-1, self.state_dim)),  # obs
                'next_obs': self.buffer_next_obs[:upper_bound, ...].reshape((-1, self.state_dim)),  # next obs
                'rewards': self.buffer_rewards[:upper_bound],  # reward
                'dones': self.buffer_dones[:upper_bound],  # done
                'terminated': self.buffer_terminated[:upper_bound]  # terminated
            }
            if self.use_feasibility:
                batch.update({
                    'ego_CBV_obs': self.buffer_ego_CBV_obs[:upper_bound],
                    'ego_CBV_next_obs': self.buffer_ego_CBV_next_obs[:upper_bound],
                    'ego_actions': self.buffer_ego_actions[:upper_bound],
                    'ego_CBV_dis': self.buffer_ego_CBV_dis[:upper_bound],
                    'ego_CBV_next_dis': self.buffer_ego_CBV_next_dis[:upper_bound]
                })
        elif self.mode == 'train_agent':
            index = 0
            actions = np.zeros((self.buffer_len, self.action_dim), dtype=np.float32)
            log_probs = np.zeros(self.buffer_len, dtype=np.float32)
            obs = np.zeros((self.buffer_len, *self.obs_shape), dtype=np.float32)
            next_obs = np.zeros((self.buffer_len, *self.obs_shape), dtype=np.float32)
            rewards = np.zeros(self.buffer_len, dtype=np.float32)
            dones = np.zeros(self.buffer_len, dtype=np.float32)
            for scenario_id in range(self.num_scenario):
                upper_bound = self.buffer_capacity // 2 if self.agent_full[scenario_id] else self.agent_pos[scenario_id]
                actions[index:index + upper_bound] = self.buffer_actions[:upper_bound, scenario_id, :]
                log_probs[index:index + upper_bound] = self.buffer_log_probs[:upper_bound, scenario_id]
                obs[index:index + upper_bound, ...] = self.buffer_obs[:upper_bound, scenario_id, ...]
                next_obs[index:index + upper_bound, ...] = self.buffer_next_obs[:upper_bound, scenario_id, ...]
                rewards[index:index + upper_bound] = self.buffer_rewards[:upper_bound, scenario_id]
                dones[index:index + upper_bound] = self.buffer_dones[:upper_bound, scenario_id]
                index += upper_bound

            batch = {
                'actions': actions,  # action
                'log_probs': log_probs,  # action log probability
                'obs': obs.reshape((-1, self.state_dim)),  # obs
                'next_obs': next_obs.reshape((-1, self.state_dim)),  # next obs
                'rewards': rewards,  # reward
                'dones': dones,  # done
            }
        else:
            raise ValueError(f'{self.mode} mode do not need to get buffer data')

        return batch

    def check_scenario_id_for_saving(self, config_len):
        if config_len == self.num_scenario:
            self.store_scenario_id = None
        else:
            if self.store_scenario_id is None:
                self.store_scenario_id = 0
            else:
                self.store_scenario_id = (self.store_scenario_id + 1) % self.num_scenario  # change the scenario id to store data

    def save_feasibility_data(self, file_path):
        assert self.mode == 'collect_feasibility_data'
        index = 0
        actions = np.zeros((self.buffer_len, self.action_dim), dtype=np.float32)
        obs = np.zeros((self.buffer_len, *self.obs_shape), dtype=np.float32)
        next_obs = np.zeros((self.buffer_len, *self.obs_shape), dtype=np.float32)
        ego_min_dis = np.zeros(self.buffer_len, dtype=np.float32)
        ego_collide = np.zeros(self.buffer_len, dtype=np.float32)
        dones = np.zeros(self.buffer_len, dtype=np.float32)

        for scenario_id in range(self.num_scenario):
            upper_bound = self.buffer_capacity // 2 if self.feasibility_full[scenario_id] else self.feasibility_pos[scenario_id]
            actions[index:index + upper_bound] = self.buffer_actions[:upper_bound, scenario_id, :]
            obs[index:index + upper_bound, ...] = self.buffer_obs[:upper_bound, scenario_id, ...]
            next_obs[index:index + upper_bound, ...] = self.buffer_next_obs[:upper_bound, scenario_id, ...]
            ego_min_dis[index:index + upper_bound] = self.buffer_ego_min_dis[:upper_bound, scenario_id]
            ego_collide[index:index + upper_bound] = self.buffer_ego_collide[:upper_bound, scenario_id]
            dones[index:index + upper_bound] = self.buffer_dones[:upper_bound, scenario_id]
            index += upper_bound

        with h5py.File(file_path, 'w') as file:
            file.attrs['length'] = self.buffer_len
            file.attrs.update({'action_dim': self.action_dim, 'obs_shape': self.obs_shape})
            file.create_dataset('actions', shape=(self.buffer_len, self.action_dim), dtype=np.float32, data=actions, compression='gzip')
            file.create_dataset('obs', shape=(self.buffer_len, *self.obs_shape), dtype=np.float32, data=obs.reshape((-1, self.state_dim)), compression='gzip')
            file.create_dataset('next_obs', shape=(self.buffer_len, *self.obs_shape), dtype=np.float32, data=next_obs.reshape((-1, self.state_dim)), compression='gzip')
            file.create_dataset('ego_min_dis', shape=self.buffer_len, dtype=np.float32, data=ego_min_dis, compression='gzip')
            file.create_dataset('ego_collide', shape=self.buffer_len, dtype=np.float32, data=ego_collide, compression='gzip')
            file.create_dataset('dones', shape=self.buffer_len, dtype=np.float32, data=dones, compression='gzip')
