#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：behavior.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
'''

import numpy as np

from safebench.agent.base_policy import BasePolicy
from safebench.agent.expert.autopilot import AutoPilot


class CarlaExpertAgent(BasePolicy):
    name = 'expert'
    type = 'unlearnable'

    """ This is just an example for testing, which always goes straight. """
    def __init__(self, config, logger):
        self.logger = logger

        self.model_path = config['model_path']
        self.mode = 'train'
        self.continue_episode = 0
        self.route = None
        self.controller_list = []

    def set_ego_and_route(self, ego_vehicles, info):
        self.ego_vehicles = ego_vehicles
        self.controller_list = []
        for e_i, ego in enumerate(ego_vehicles):
            controller = AutoPilot(ego)
            gps_route = info[e_i]['gps_route']  # the gps route
            route = info[e_i]['route']  # the world coord route
            controller.set_planner(gps_route, route)  # set route for each controller

            self.controller_list.append(controller)

    def train(self, replay_buffer):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, obs, infos, deterministic=False):
        actions = []
        for e_i in infos:
            # select the controller that matches the scenario_id
            control = self.controller_list[e_i['scenario_id']].run_step()
            throttle = control.throttle
            steer = control.steer
            brake = control.brake
            actions.append([throttle, steer, brake])
        actions = np.array(actions, dtype=np.float32)
        return actions

    def load_model(self):
        pass

    def save_model(self):
        pass
