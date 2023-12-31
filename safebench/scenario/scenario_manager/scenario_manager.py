#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：scenario_manager.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_manager.timer import GameTime
from safebench.scenario.tools.scenario_utils import calculate_distance_locations


class ScenarioManager(object):
    """
        Dynamic version scenario manager class. This class holds all functionality
        required to initialize, trigger, update and stop a scenario.
    """

    def __init__(self, env_params, logger, use_scenic=False):
        self.env_params = env_params
        self.logger = logger
        self.scenic = use_scenic
        self._collision = False
        self.collide_with_cbv = False
        self.truncated = False
        self.cbv = None
        self._reset()

    def _reset(self):
        #self.scenario = None
        self.route_scenario = None
        self.ego_vehicle = None
        self.cbv = None
        self._running = False
        self._collision = False
        self.collide_with_cbv = False
        self.truncated = False
        self._timestamp_last_run = 0.0
        self.running_record = []
        GameTime.restart()

    def clean_up(self):
        if self.route_scenario is not None:
            self.route_scenario.clean_up()

    def load_scenario(self, scenario):
        self._reset()
        self.route_scenario = scenario  # the RouteScenario
        self.ego_vehicle = scenario.ego_vehicle
        self.scenario_instance = scenario.scenario_instance  # the adv behavior single scenario instance

    def update_cbv_nearby_vehicles(self, cbv, cbv_nearby_vehicles):
        self.cbv = cbv
        self.cbv_nearby_vehicles = cbv_nearby_vehicles
        self.route_scenario.cbv = cbv
        self.route_scenario.cbv_nearby_vehicles = cbv_nearby_vehicles

    def run_scenario(self):
        self._running = True
        self._init_scenarios()  # generate the background vehicle

    def _init_scenarios(self):
        # spawn route actors for each scenario
        self.route_scenario.initialize_actors()  # generate the background vehicle
    
    def stop_scenario(self):
        self._running = False

    def update_running_status(self):
        record, stop, collision, collide_with_cbv, truncated = self.route_scenario.get_running_status(self.running_record)
        self.running_record.append(record)  # contain every step's record
        if stop:
            self._running = False
        if collision:
            self._collision = True
            if collide_with_cbv:
                self.collide_with_cbv = True
        if truncated:
            self.truncated = True

    def get_update(self, timestamp, scenario_action):
        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds
            GameTime.on_carla_tick(timestamp)
            # update the scenario instance receiving the scenario action
            self.scenario_instance.update_behavior(self.cbv, scenario_action)
