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
        self.acc_range = config['acc_range']
        self.steer_range = config['steer_range']

    def set_ego_and_route(self, ego_vehicles, info):
        self.ego_vehicles = ego_vehicles

    def train(self, replay_buffer, writer, e_i):
        pass

    def set_mode(self, mode):
        pass

    def get_action(self, obs, infos, deterministic=False):
        actions = []
        for _ in infos:

            acc = np.random.uniform(self.acc_range[0], self.acc_range[1])
            steer = np.random.uniform(self.steer_range[0], self.steer_range[1])

            # normalize and clip the action
            acc = acc * self.acc_range[1]
            steer = steer * self.steer_range[1]
            acc = max(min(self.acc_range[1], acc), self.acc_range[0])
            steer = max(min(self.steer_range[1], steer), self.steer_range[0])

            # Convert acceleration to throttle and brake
            if acc > 0:
                throttle = np.clip(acc / 3, 0, 1)
                brake = 0
            else:
                throttle = 0
                brake = np.clip(-acc / 8, 0, 1)
            actions.append([throttle, steer, brake])
        actions = np.array(actions, dtype=np.float32)
        return actions

    def load_model(self, map_name):
        pass

    def save_model(self, episode, map_name, buffer):
        pass
