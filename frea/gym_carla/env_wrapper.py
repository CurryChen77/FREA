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
import carla
import copy
import numpy as np
import pygame

from frea.scenario.scenario_manager.carla_data_provider import CarlaDataProvider


class VectorWrapper():
    """ 
        The interface to control a list of environments.
    """

    def __init__(self, env_params, scenario_config, world, birdeye_render, display, use_feasibility, agent_state_encoder, logger):
        self.logger = logger
        self.world = world
        self.num_scenario = scenario_config['num_scenario']  # default 2
        self.ROOT_DIR = scenario_config['ROOT_DIR']
        self.frame_skip = scenario_config['frame_skip']
        self.spectator = scenario_config['spectator']
        self.agent_state_encoder = agent_state_encoder
        self.birdeye_render = birdeye_render
        self.mode = env_params['mode']
        self.eval_mode = env_params['eval_mode']

        self.env_list = []
        for i in range(self.num_scenario):
            # each small scenario corresponds to a carla_env create the ObservationWrapper()
            env = carla_env(
                env_params, birdeye_render=birdeye_render, display=display,
                world=world, use_feasibility=use_feasibility,
                agent_state_encoder=agent_state_encoder, logger=logger
            )
            self.env_list.append(env)

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

        CarlaDataProvider.on_carla_tick()  # tick since each small scenario got several warm-up ticks

        if self.spectator:
            transform = CarlaDataProvider.get_first_ego_transform()  # from the first ego vehicle view
            if transform is not None:
                spectator = self.world.get_spectator()
                spectator.set_transform(carla.Transform(
                    transform.location + carla.Location(x=-3, z=50), carla.Rotation(yaw=transform.rotation.yaw, pitch=-80.0)
                ))

        # sometimes not all scenarios are used
        self.finished_env = [False] * self.num_scenario
        for s_i in range(len(scenario_configs), self.num_scenario):
            self.finished_env[s_i] = True

        # store scenario id
        for s_i in range(len(scenario_configs)):
            info_list[s_i].update({'scenario_id': s_i})

        # return obs
        return self.obs_postprocess(obs_list), info_list

    def step(self, ego_actions, scenario_actions, onpolicy):
        """
            ego_actions: [num_alive_scenario]
            scenario_actions: [num_alive_scenario]
        """
        if onpolicy['scenario']:
            # the onpolicy scenario actions [actions, log_probs]
            scenario_actions = scenario_actions[0]
        if onpolicy['agent']:
            # the onpolicy agent scenario actions [actions, log_probs]
            ego_actions = ego_actions[0]

        # apply action
        action_idx = 0  # action idx should match the env that is not finished
        for e_i in range(self.num_scenario):
            if not self.finished_env[e_i]:
                processed_action = self.env_list[e_i]._postprocess_action(ego_actions[action_idx])
                self.env_list[e_i].step_before_tick(processed_action, scenario_actions[action_idx])

                action_idx += 1

        if self.spectator:
            transform = CarlaDataProvider.get_first_ego_transform()  # from the first ego vehicle view
            if transform is not None:
                spectator = self.world.get_spectator()
                spectator.set_transform(carla.Transform(
                    transform.location + carla.Location(x=-3, z=50), carla.Rotation(yaw=transform.rotation.yaw, pitch=-80.0)
                ))

        # tick all scenarios
        for _ in range(self.frame_skip):
            self.world.tick()

        # collect new observation of one frame
        train_obs_list = []
        transition_obs_list = []
        reward_list = []
        done_list = []
        train_info_list = []
        transition_info_list = []
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
                    # save running results according to the data_id of scenario
                    if self.eval_mode == 'analysis':
                        if current_env.config.data_id in self.running_results.keys():
                            self.logger.log('Scenario with data_id {} is duplicated'.format(current_env.config.data_id))
                        # the running results contain every data id (one specific scenario) running status at each time step
                        self.running_results[current_env.config.data_id] = current_env.scenario_manager.running_record
                else:
                    # if the scenario is done, no need to do the transition
                    transition_obs_list.append(obs)
                    transition_info_list.append(info[1])

                # update information
                train_obs_list.append(obs)
                reward_list.append(reward)
                done_list.append(done)
                train_info_list.append(info[0])
        
        # convert to numpy
        rewards = np.array(reward_list, dtype=np.float32)
        dones = np.array(done_list, dtype=np.float32)
        train_infos = np.array(train_info_list)
        transition_infos = np.array(transition_info_list)

        # update pygame window
        if self.eval_mode == 'render' and self.birdeye_render:
            pygame.display.flip()
        return self.obs_postprocess(train_obs_list), self.obs_postprocess(transition_obs_list), rewards, dones, train_infos, transition_infos

    def all_scenario_done(self):
        if np.sum(self.finished_env) == self.num_scenario:
            return True
        else:
            return False

    def clean_up(self):
        # clean the temp list in the render
        self.birdeye_render.clean_up() if self.birdeye_render else None
        # stop sensor objects
        for e_i in range(self.num_scenario):
            self.env_list[e_i].clean_up()

        # clean the CarlaDataProvider
        CarlaDataProvider.clean_up_after_episode()
        # tick to ensure that all destroy commands are executed
        self.world.tick()


class ObservationWrapper(gym.Wrapper):
    """
        The wrapped carla environment.
    """
    def __init__(self, env, agent_obs_type):
        super().__init__(env)
        self._env = env  # carla environment

        self.agent_obs_type = agent_obs_type

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        return self._preprocess_obs(obs), info

    def step_before_tick(self, ego_action, scenario_action):
        self._env.step_before_tick(ego_action=ego_action, scenario_action=scenario_action)

    def step_after_tick(self):
        obs, reward, done, info = self._env.step_after_tick()
        reward, info = self._preprocess_reward(reward, info)
        obs = self._preprocess_obs(obs)
        return obs, reward, done, info

    def _preprocess_obs(self, obs):
        if self.agent_obs_type == 'ego_state':
            process_obs = obs['ego_state']  # include the pos, speed, compass(yaw angle)
        elif self.agent_obs_type == 'ego_obs':
            process_obs = obs['ego_obs'].flatten()  # flatten the 2D state space
        elif self.agent_obs_type == 'no_obs':
            process_obs = obs
        else:
            raise NotImplementedError

        return process_obs

    def _preprocess_reward(self, reward, info):
        return reward, info

    def _postprocess_action(self, action):
        return action

    def clean_up(self):
        self._env.clean_up()


def carla_env(env_params, birdeye_render=None, display=None, world=None, use_feasibility=None, agent_state_encoder=None, logger=None):
    return ObservationWrapper(
        gym.make(
            'carla-v0', 
            env_params=env_params, 
            birdeye_render=birdeye_render,
            display=display, 
            world=world,
            use_feasibility=use_feasibility,
            agent_state_encoder=agent_state_encoder,
            logger=logger,
        ), 
        agent_obs_type=env_params['agent_obs_type'],
    )
