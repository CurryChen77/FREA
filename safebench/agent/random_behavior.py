#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：behavior.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""

import numpy as np

from safebench.agent.base_policy import BasePolicy
from safebench.gym_carla.envs.utils import linear_map


class RandomAgent(BasePolicy):
    name = 'random_policy'
    type = 'unlearnable'

    """ This is just an random agent for offline data collection. """
    def __init__(self, config, logger):
        self.logger = logger
        self.num_scenario = config['num_scenario']
        self.throttle_range = config['throttle_range']
        self.brake_range = config['brake_range']
        self.steer_range = config['steer_range']

    def set_ego_and_route(self, ego_vehicles, info):
        self.ego_vehicles = ego_vehicles

    def train(self, replay_buffer, writer, e_i):
        pass

    def set_mode(self, mode):
        pass

    def get_action(self, obs, infos, deterministic=False):
        actions = []
        for e_i in infos:

            throttle = np.random.uniform(self.throttle_range[0], self.throttle_range[1])
            steer = np.random.uniform(self.steer_range[0], self.steer_range[1])
            brake = np.random.uniform(self.brake_range[0], self.brake_range[1])

            if brake < 0.05:
                brake = 0.0
            if throttle > brake:
                brake = 0.0

            actions.append([throttle, steer, brake])
        actions = np.array(actions, dtype=np.float32)
        return actions

    def load_model(self, map_name):
        pass

    def save_model(self, episode, map_name, buffer):
        pass
