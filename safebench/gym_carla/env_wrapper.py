#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：env_wrapper.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""

import gym
import numpy as np
import pygame
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider


class VectorWrapper():
    """ 
        The interface to control a list of environments.
    """

    def __init__(self, env_params, scenario_config, world, birdeye_render, display, search_radius, agent_state_encoder, logger):
        self.logger = logger
        self.world = world
        self.num_scenario = scenario_config['num_scenario']  # default 2
        self.ROOT_DIR = scenario_config['ROOT_DIR']
        self.frame_skip = scenario_config['frame_skip']  
        self.render = scenario_config['render']
        self.agent_state_encoder = agent_state_encoder

        self.env_list = []
        self.action_space_list = []
        for i in range(self.num_scenario):
            # each small scenario corresponds to a carla_env create the ObservationWrapper()
            env = carla_env(
                env_params, birdeye_render=birdeye_render, display=display,
                world=world, search_radius=search_radius, agent_state_encoder=agent_state_encoder, logger=logger)
            self.env_list.append(env)
            self.action_space_list.append(env.action_space)

        # flags for env list 
        self.finished_env = [False] * self.num_scenario
        self.running_results = {}
    
    def obs_postprocess(self, obs_list):
        # assume all variables are array
        obs_list = np.array(obs_list)
        return obs_list

    def get_ego_vehicles(self):
        ego_vehicles = []
        for env in self.env_list:
            if env.ego_vehicle is not None:
                # self.logger.log('>> Ego vehicle is None. Please call reset() first.', 'red')
                # raise Exception()
                ego_vehicles.append(env.ego_vehicle)
        return ego_vehicles

    def reset(self, scenario_configs):
        # create scenarios and ego vehicles
        obs_list = []
        info_list = []
        for s_i in range(len(scenario_configs)):
            # for each scenario in the town
            config = scenario_configs[s_i]
            obs, info = self.env_list[s_i].reset(
                config=config,
                env_id=s_i,
                # scenario_init_action=scenario_init_action[s_i]
                )
            obs_list.append(obs)
            info_list.append(info)

        # sometimes not all scenarios are used
        self.finished_env = [False] * self.num_scenario
        for s_i in range(len(scenario_configs), self.num_scenario):
            self.finished_env[s_i] = True
        
        # store scenario id
        for s_i in range(len(scenario_configs)):
            info_list[s_i].update({'scenario_id': s_i})

        # return obs
        return self.obs_postprocess(obs_list), info_list

    def step(self, ego_actions, scenario_actions):
        """
            ego_actions: [num_alive_scenario]
            scenario_actions: [num_alive_scenario]
        """
        # apply action
        action_idx = 0  # action idx should match the env that is not finished
        for e_i in range(self.num_scenario):
            if not self.finished_env[e_i]:
                processed_action = self.env_list[e_i]._postprocess_action(ego_actions[action_idx])
                # TODO: pre-process scenario action
                self.env_list[e_i].step_before_tick(processed_action, scenario_actions[action_idx])
                action_idx += 1
        
        # tick all scenarios
        for _ in range(self.frame_skip):
            self.world.tick()

        # collect new observation of one frame
        obs_list = []
        reward_list = []
        done_list = []
        info_list = []
        for e_i in range(self.num_scenario):
            if not self.finished_env[e_i]:
                current_env = self.env_list[e_i]
                obs, reward, done, info = current_env.step_after_tick()

                # store scenario id to help agent decide which policy should be used
                info[0]['scenario_id'] = e_i
                info[1]['scenario_id'] = e_i

                # check if env is done
                if done:
                    self.finished_env[e_i] = True
                    if current_env.scenario_manager._collision:
                        ego_min_dis = CarlaDataProvider.get_ego_min_dis()
                        self.logger.log(f'>> collision caused ego min dis = {ego_min_dis}', color='yellow')
                    # save running results according to the data_id of scenario
                    if current_env.config.data_id in self.running_results.keys():
                        self.logger.log('Scenario with data_id {} is duplicated'.format(current_env.config.data_id))
                    self.running_results[current_env.config.data_id] = current_env.scenario_manager.running_record

                # update infomation
                obs_list.append(obs)
                reward_list.append(reward)
                done_list.append(done)
                info_list.append(info)
        
        # convert to numpy
        rewards = np.array(reward_list)
        dones = np.array(done_list)
        infos = np.array(info_list)

        # update pygame window
        if self.render:
            pygame.display.flip()
        return self.obs_postprocess(obs_list), rewards, dones, infos

    # def sample_action_space(self):
    #     action = []
    #     for action_space in self.action_space_list:
    #         action.append(action_space.sample())
    #     return np.array(action)

    def all_scenario_done(self):
        if np.sum(self.finished_env) == self.num_scenario:
            return True
        else:
            return False

    # def set_background_vehicles(self, background_vehicles):
    #     self.background_vehicles = background_vehicles
    #
    # def clean_background_vehicles(self):
    #     # remove background vehicles
    #     self.logger.log(f'>> cleaning {len(self.background_vehicles)} backgound_agents', color='green')
    #     for s_i in range(len(self.background_vehicles)):
    #         if self.background_vehicles[s_i].type_id.startswith('vehicle'):
    #             self.background_vehicles[s_i].set_autopilot(enabled=False)
    #         if CarlaDataProvider.actor_id_exists(self.background_vehicles[s_i].id):
    #             CarlaDataProvider.remove_actor_by_id(self.background_vehicles[s_i].id)

    def clean_up(self):
        # stop sensor objects
        for e_i in range(self.num_scenario):
            self.env_list[e_i].clean_up()

        # tick to ensure that all destroy commands are executed
        self.world.tick()


class ObservationWrapper(gym.Wrapper):
    """
        The wrapped carla environment.
    """
    def __init__(self, env, agent_obs_type, safety_network_obs_type):
        super().__init__(env)
        self._env = env  # carla environment

        self.is_running = False
        self.agent_obs_type = agent_obs_type
        self.safety_network_obs_type = safety_network_obs_type
        # self._build_obs_space()

        # build action space, assume the obs range from -1 to 1
        act_dim = 2
        act_lim = np.ones((act_dim), dtype=np.float32)
        self.action_space = gym.spaces.Box(-act_lim, act_lim, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        return self._preprocess_obs(obs), info

    def step_before_tick(self, ego_action, scenario_action):
        self._env.step_before_tick(ego_action=ego_action, scenario_action=scenario_action)

    def step_after_tick(self):
        obs, reward, done, info = self._env.step_after_tick()
        self.is_running = self._env.is_running
        reward, info = self._preprocess_reward(reward, info)
        obs = self._preprocess_obs(obs)
        return obs, reward, done, info

    # def _build_obs_space(self):
    #     if self.obs_type == 0:
    #         obs_dim = 4
    #         # assume the obs range from -1 to 1
    #         obs_lim = np.ones((obs_dim), dtype=np.float32)
    #         self.observation_space = gym.spaces.Box(-obs_lim, obs_lim)
    #     elif self.obs_type == 1:
    #         obs_dim = 11
    #         # assume the obs range from -1 to 1
    #         obs_lim = np.ones((obs_dim), dtype=np.float32)
    #         self.observation_space = gym.spaces.Box(-obs_lim, obs_lim)
    #     elif self.obs_type == 2 or self.obs_type == 3:
    #         # 4 state space + bev
    #         obs_dim = 128  # TODO: should be the same as display_size
    #         # assume the obs range from -1 to 1
    #         obs_lim = np.ones((obs_dim), dtype=np.float32)
    #         self.observation_space = gym.spaces.Box(-obs_lim, obs_lim)
    #     else:
    #         raise NotImplementedError

    def _preprocess_obs(self, obs):
        # only use the 4-dimensional state space
        if self.agent_obs_type == 'simple_state':
            return obs['simple_state'][:4].astype(np.float64)
        # include the pos, speed, compass(yaw angle)
        elif self.agent_obs_type == 'ego_state':
            return obs['ego_state']
        elif self.agent_obs_type == 'no_obs':
            return None
        else:
            raise NotImplementedError

    def _preprocess_reward(self, reward, info):
        return reward, info

    def _postprocess_action(self, action):
        return action

    def clear_up(self):
        self._env.clear_up()


def carla_env(env_params, birdeye_render=None, display=None, world=None, search_radius=0, agent_state_encoder=None, logger=None):
    return ObservationWrapper(
        gym.make(
            'carla-v0', 
            env_params=env_params, 
            birdeye_render=birdeye_render,
            display=display, 
            world=world,
            search_radius=search_radius,
            agent_obs_type=env_params['agent_obs_type'],
            agent_state_encoder=agent_state_encoder,
            logger=logger,
        ), 
        agent_obs_type=env_params['agent_obs_type'],
        safety_network_obs_type=agent_state_encoder.obs_type if agent_state_encoder else None
    )
