#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：expert.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/22
@source  ：This project is modified from <https://github.com/autonomousvision/plant/tree/1bfb695910d816e70f53521aa263648072edea8e>
"""

import numpy as np

from safebench.agent.base_policy import BasePolicy
from safebench.agent.expert.autopilot import AutoPilot


class CarlaExpertAgent(BasePolicy):
    name = 'expert'
    type = 'unlearnable'

    """ This is just an example for testing, which always goes straight. """
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.model_path = config['model_path']
        self.viz_route = config['viz_route']
        self.mode = 'train'
        self.continue_episode = 0
        self.route = None
        self.controller_list = []
        for _ in range(config['num_scenario']):
            controller = AutoPilot(self.config, self.logger)  # initialize the controller
            self.controller_list.append(controller)

    def set_ego_and_route(self, ego_vehicles, info):
        self.ego_vehicles = ego_vehicles
        for i, ego in enumerate(ego_vehicles):
            gps_route = info[i]['gps_route']  # the gps route
            route = info[i]['route']  # the world coord route
            self.controller_list[i].set_planner(ego, gps_route, route)  # set route for each controller

    def train(self, replay_buffer):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, obs, infos, deterministic=False):
        actions = []
        for i, e_i in enumerate(infos):
            # select the controller that matches the scenario_id
            control = self.controller_list[e_i['scenario_id']].run_step(obs[i], self.viz_route)
            throttle = control.throttle
            steer = control.steer
            brake = control.brake
            actions.append([throttle, steer, brake])
        actions = np.array(actions, dtype=np.float32)
        return actions

    def load_model(self, map_name):
        pass

    def save_model(self, episode, map_name):
        pass