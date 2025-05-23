#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：adv_behavior_scenario.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""

import carla

import numpy as np
from frea.scenario.tools.scenario_operation import ScenarioOperation
from frea.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from frea.scenario.scenario_definition.basic_scenario import BasicScenario
from frea.scenario.tools.scenario_utils import calculate_distance_transforms


class AdvBehaviorSingle(BasicScenario):
    """
        This class holds everything required for a scenario, in which an other vehicle takes priority from the ego vehicle, 
        by running a red traffic light (while the ego vehicle has green).
    """

    def __init__(self, world, ego_vehicle, env_params, timeout=100):
        super(AdvBehaviorSingle, self).__init__("AdvBehaviorSingle", None, world)
        self.timeout = timeout
        self.ego_vehicle = ego_vehicle
        self.signalized_junction = env_params['signalized_junction']
        if self.signalized_junction:
            self._map = CarlaDataProvider.get_map()
            self.last_ego_waypoint = self._map.get_waypoint(self.ego_vehicle.get_location())
            self.traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
            if self.traffic_light is None:
                print(">> No traffic light for the given location of the ego vehicle found")
            else:
                self.traffic_light.set_state(carla.TrafficLightState.Green)
                self.traffic_light.set_green_time(self.timeout)
        else:
            # set all the traffic light to green
            CarlaDataProvider.set_all_traffic_light(traffic_light_state=carla.TrafficLightState.Green, timeout=self.timeout)

        self.acc_max = env_params['continuous_accel_range'][1]
        self.steering_max = env_params['continuous_steer_range'][1]

    def convert_actions(self, scenario_actions):
        acc = scenario_actions[0]  # continuous action: acc
        steer = scenario_actions[1]  # continuous action: steering

        # normalize and clip the action
        acc = acc * self.acc_max
        steer = steer * self.steering_max
        acc = max(min(self.acc_max, acc), -self.acc_max)
        steer = max(min(self.steering_max, steer), -self.steering_max)

        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc / 3, 0, 1)
            brake = 0
            reverse = False
        else:
            # throttle = 0
            # brake = np.clip(-acc / 8, 0, 1)
            # reverse = False
            # enable driving back
            reverse = True
            throttle = -np.clip(acc / 3, -1, 0)
            brake = 0

        # apply ego control
        act = carla.VehicleControl(reverse=reverse, throttle=float(throttle), steer=float(steer), brake=float(brake))
        return act

    def update_traffic_light(self):
        ego_waypoint = self._map.get_waypoint(CarlaDataProvider.get_location(self.ego_vehicle))
        if not ego_waypoint.is_junction and self.last_ego_waypoint.is_junction:  # last tick the ego is in the junction, but the current step is out
            traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle)
            # if the ego's next traffic light is not None and has changed, then set the next traffic light to green
            if traffic_light is not None and traffic_light != self.traffic_light:
                self.traffic_light = traffic_light
                traffic_light.set_state(carla.TrafficLightState.Green)
                traffic_light.set_green_time(self.timeout)
        self.last_ego_waypoint = ego_waypoint

    def update_behavior(self, scenario_actions):
        if scenario_actions is not None:
            # apply scenario action for each CBV
            for CBV_id, CBV in CarlaDataProvider.get_CBVs_by_ego(self.ego_vehicle).items():
                scenario_action = scenario_actions[CBV_id]
                act = self.convert_actions(scenario_action)
                CBV.apply_control(act)  # apply the control of the CBV on the next tick

        if self.signalized_junction:  # if the signal controls the junction, the traffic need to be updated
            self.update_traffic_light()

    def clean_up(self):
        pass
