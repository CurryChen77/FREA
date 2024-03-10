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


class CarlaExpertDisturbAgent(BasePolicy):
    name = 'expert_disturb'
    type = 'unlearnable'

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.model_path = config['model_path']
        self.viz_route = config['viz_route']
        self.throttle_disturb = config['throttle_disturb']
        self.steer_disturb = config['steer_disturb']
        self.brake_disturb = config['brake_disturb']
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

    def train(self, replay_buffer, writer, e_i):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, obs, infos, deterministic=False):
        actions = []
        for i, e_i in enumerate(infos):
            throttle_disturb = np.random.uniform(self.throttle_disturb[0], self.throttle_disturb[1])
            steer_disturb = np.random.uniform(self.steer_disturb[0], self.steer_disturb[1])
            brake_disturb = np.random.uniform(self.brake_disturb[0], self.brake_disturb[1])
            # select the controller that matches the scenario_id
            control = self.controller_list[e_i['scenario_id']].run_step(obs[i], self.viz_route)
            # when the expert throttle is not 0, disturb the throttle level
            throttle = min(max(control.throttle + throttle_disturb, 0.01), 1) if control.throttle > 0.01 else control.throttle
            # disturb steer
            steer = min(max(control.steer + steer_disturb, -1), 1)
            # when expert brake is not 0, disturb the brake level
            brake = min(max(control.brake + brake_disturb, 0.01), 1) if control.brake > 0.01 else control.brake
            actions.append([throttle, steer, brake])
        actions = np.array(actions, dtype=np.float32)
        return actions

    def load_model(self, map_name):
        pass

    def save_model(self, episode, map_name, buffer):
        pass