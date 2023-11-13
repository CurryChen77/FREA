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

        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicle, False)
        if self._traffic_light is None:
            print(">> No traffic light for the given location of the ego vehicle found")
        else:
            self._traffic_light.set_state(carla.TrafficLightState.Green)
            self._traffic_light.set_green_time(self.timeout)

        self.scenario_operation = ScenarioOperation()
        self.trigger = False
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
        traffic_light = CarlaDataProvider.set_vehicle_next_traffic_light(self.ego_vehicle)
        # if the ego's next traffic light is not None and has changed, then set the next traffic light to green
        if traffic_light is not None and traffic_light != self._traffic_light:
            self._traffic_light = traffic_light
            print("set the next traffic light to green")
            traffic_light.set_state(carla.TrafficLightState.Green)
            traffic_light.set_green_time(self.timeout)

    def create_behavior(self, scenario_init_action):
        assert scenario_init_action is None, f'{self.name} should receive [None] initial action.'
        self.other_actor_delta_x = 1.0
        self.trigger_distance_threshold = 35

    def update_behavior(self, controlled_bv, scenario_action):
        # update the traffic light
        self.update_traffic_light()
        # if the controlled bv exists and the scenario policy isn't hardcoded
        if controlled_bv is not None and scenario_action is not None:
            # create the action
            act = self.convert_actions(scenario_action)
            if self.prior_controlled_bv is None:  # the initial time
                controlled_bv.set_autopilot(enabled=False)  # get ready to be controlled
                self.prior_controlled_bv = controlled_bv
            else:
                if self.prior_controlled_bv != controlled_bv:  # the controlled bv has changed
                    self.prior_controlled_bv.set_autopilot(enabled=True)  # activate the autopilot mode of prior bv
                    controlled_bv.set_autopilot(enabled=False)  # get ready to be controlled
                    self.prior_controlled_bv = controlled_bv  # update the prior controlled bv

            self.prior_controlled_bv.apply_control(act)  # apply the control of the cbv on the next tick

    def check_stop_condition(self):
        # stop when actor runs a specific distance
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.actor_transform_list[0])
        if cur_distance >= self._actor_distance:
            return True
        return False
