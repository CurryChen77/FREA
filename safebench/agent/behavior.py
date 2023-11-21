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
from agents.navigation.behavior_agent import BehaviorAgent
from safebench.agent.agent_utils.visualization import draw_route


class CarlaBehaviorAgent(BasePolicy):
    name = 'behavior'
    type = 'unlearnable'

    """ This is just an example for testing, which always goes straight. """
    def __init__(self, config, logger):
        self.logger = logger
        self.num_scenario = config['num_scenario']
        self.model_path = config['model_path']
        self.viz_route = config['viz_route']
        self.mode = 'train'
        self.continue_episode = 0
        self.route = None
        self.controller_list = []
        behavior_list = ["cautious", "normal", "aggressive"]
        self.behavior = behavior_list[1]

    def set_ego_and_route(self, ego_vehicles, info):
        self.ego_vehicles = ego_vehicles
        self.controller_list = []
        for e_i in range(len(ego_vehicles)):
            controller = BehaviorAgent(self.ego_vehicles[e_i], behavior=self.behavior)
            dest_waypoint = info[e_i]['route_waypoints'][-1]  # the destination of the ego vehicle
            location = dest_waypoint.transform.location
            controller.set_destination(location)  # set route for each controller
            self.controller_list.append(controller)

    def train(self, replay_buffer):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, obs, infos, deterministic=False):
        actions = []
        for e_i in infos:
            controller = self.controller_list[e_i['scenario_id']]

            if self.viz_route:
                route = controller._local_planner._waypoints_queue
                waypoint_route = np.array([[node[0].transform.location.x, node[0].transform.location.y] for node in route])
                if len(waypoint_route) >= 1:
                    draw_route(controller._world, controller._vehicle, waypoint_route)

            # TODO the waypoint list in safebench and carla's behavior agent is different
            # for the behavior agent, the goal may be reached (no more waypoints to chase), but safebench still got waypoints
            if controller.done():
                throttle = 0
                steer = 0
                brake = 0.2
            else:
                # select the controller that matches the scenario_id
                control = controller.run_step(debug=False)
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
