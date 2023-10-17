#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：scenario_manager.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
'''

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
        self._reset()

    def _reset(self):
        #self.scenario = None
        self.route_scenario = None
        self.ego_vehicle = None
        self.scenario_list = None
        self.triggered_scenario = set()
        self._running = False
        self._collision = False
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

    def update_controlled_bv_nearby_vehicles(self, controlled_bv, controlled_bv_nearby_vehicles):
        self.controlled_bv = controlled_bv
        self.controlled_bv_nearby_vehicles = controlled_bv_nearby_vehicles
        self.route_scenario.controlled_bv = controlled_bv
        self.route_scenario.controlled_bv_nearby_vehicles = controlled_bv_nearby_vehicles

    def run_scenario(self):
        self._running = True
        self._init_scenarios()  # generate the background vehicle

    def _init_scenarios(self):
        # spawn route actors for each scenario
        self.route_scenario.initialize_actors()  # generate the background vehicle
    
    def stop_scenario(self):
        self._running = False

    def update_running_status(self):
        record, stop, collision = self.route_scenario.get_running_status(self.running_record)
        self.running_record.append(record)
        if stop:
            self._running = False
            if collision:
                self._collision = True

    def get_update(self, timestamp, scenario_action):
        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds
            GameTime.on_carla_tick(timestamp)
            # update the CarlaDateProvider information before tick (the state of the previous time step)
            CarlaDataProvider.on_carla_tick()

            # update the scenario instance receiving the scenario action
            self.scenario_instance.update_behavior(self.controlled_bv, scenario_action)

            self.update_running_status()