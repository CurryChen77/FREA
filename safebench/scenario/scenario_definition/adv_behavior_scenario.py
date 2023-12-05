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
from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.tools.scenario_utils import calculate_distance_transforms


class AdvBehaviorSingle(BasicScenario):
    """
        This class holds everything required for a scenario, in which an other vehicle takes priority from the ego vehicle, 
        by running a red traffic light (while the ego vehicle has green).
    """

    def __init__(self, world, ego_vehicle, env_params, timeout=60):
        super(AdvBehaviorSingle, self).__init__("AdvBehaviorSingle", None, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout
        self._map = CarlaDataProvider.get_map()
        self.last_tick_affected_by_traffic = self.ego_vehicle

        self.traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        self.cbv_traffic_light = None
        self.last_ego_waypoint = self._map.get_waypoint(self.ego_vehicle.get_location())
        if self.traffic_light is None:
            print(">> No traffic light for the given location of the ego vehicle found")
        else:
            self.traffic_light.set_state(carla.TrafficLightState.Green)
            self.traffic_light.set_green_time(self.timeout)

        self._actor_distance = 110
        self.ego_max_driven_distance = 150
        self.discrete = env_params['discrete']
        self.discrete_act = [env_params['discrete_acc'], env_params['discrete_steer']]  # acc, steer
        self.n_acc = len(self.discrete_act[0])
        self.n_steer = len(self.discrete_act[1])
        self.acc_max = env_params['continuous_accel_range'][1]
        self.steering_max = env_params['continuous_steer_range'][1]
        self.prior_controlled_bv = None

    def convert_actions(self, scenario_actions):
        if self.discrete:
            acc = self.discrete_act[0][scenario_actions // self.n_steer]  # 'discrete_acc': [-3.0, 0.0, 3.0]
            steer = self.discrete_act[1][scenario_actions % self.n_steer]  # 'discrete_steer': [-0.2, 0.0, 0.2]
        else:
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
        else:
            throttle = 0
            brake = np.clip(-acc / 8, 0, 1)

        # apply ego control
        act = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
        return act

    def update_traffic_light(self):
        ego_waypoint = self._map.get_waypoint(CarlaDataProvider.get_location_after_tick(self.ego_vehicle))
        if not ego_waypoint.is_junction and self.last_ego_waypoint.is_junction:  # last tick the ego is in the junction, but the current step is out
            traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle)
            # if the ego's next traffic light is not None and has changed, then set the next traffic light to green
            if traffic_light is not None and traffic_light != self.traffic_light:
                self.traffic_light = traffic_light
                traffic_light.set_state(carla.TrafficLightState.Green)
                traffic_light.set_green_time(self.timeout)
        elif ego_waypoint.is_junction:  # if ego is in the junction and the cbv is stuck by the traffic light, set that traffic light to green
            if self.prior_controlled_bv and self.prior_controlled_bv.is_at_traffic_light:
                cbv_traffic_light = self.prior_controlled_bv.get_traffic_light()
                if cbv_traffic_light and cbv_traffic_light != self.cbv_traffic_light and cbv_traffic_light.state != carla.TrafficLightState.Green:
                    # for visualization
                    # base_transform = cbv_traffic_light.get_transform()
                    # cbv_traffic_light_loc = base_transform.transform(cbv_traffic_light.trigger_volume.location)
                    # self.world.debug.draw_point(cbv_traffic_light_loc + carla.Location(z=4), size=0.1, life_time=-1)
                    # print("set cbv next traffic light to green")
                    self.cbv_traffic_light = cbv_traffic_light
                    cbv_traffic_light.set_state(carla.TrafficLightState.Green)
                    cbv_traffic_light.set_green_time(self.timeout)
        self.last_ego_waypoint = ego_waypoint

    def update_behavior(self, controlled_bv, scenario_action):
        # if the controlled bv exists and the scenario policy isn't hardcoded
        if controlled_bv is not None and scenario_action is not None:
            if self.prior_controlled_bv is None:  # the initial time
                controlled_bv.set_autopilot(enabled=False)  # get ready to be controlled
                self.prior_controlled_bv = controlled_bv
            else:
                if self.prior_controlled_bv != controlled_bv:  # the controlled bv has changed
                    self.prior_controlled_bv.set_autopilot(enabled=True)  # activate the autopilot mode of prior bv
                    controlled_bv.set_autopilot(enabled=False)  # get ready to be controlled
                    self.prior_controlled_bv = controlled_bv  # update the prior controlled bv
            act = self.convert_actions(scenario_action)
            self.prior_controlled_bv.apply_control(act)  # apply the control of the cbv on the next tick
        elif controlled_bv is not None and scenario_action is None:
            if self.prior_controlled_bv is None:  # the initial time
                self.prior_controlled_bv = controlled_bv
            else:
                if self.prior_controlled_bv != controlled_bv:  # the controlled bv has changed
                    self.prior_controlled_bv = controlled_bv  # update the prior controlled bv

        # update the traffic light
        self.update_traffic_light()

    def clean_up(self):
        pass