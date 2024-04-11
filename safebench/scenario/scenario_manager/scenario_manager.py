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


class ScenarioManager(object):
    """
        Dynamic version scenario manager class. This class holds all functionality
        required to initialize, trigger, update and stop a scenario.
    """

    def __init__(self, env_params, logger):
        self.env_params = env_params
        self.logger = logger
        self.ego_collision = False
        self.ego_truncated = False
        self.running = False
        self.CBVs = None
        self._reset()

    def _reset(self):
        #self.scenario = None
        self.route_scenario = None
        self.ego_vehicle = None
        self.CBVs = None
        self.running = False
        self.ego_collision = False
        self.ego_truncated = False
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

    def update_CBV_nearby_vehicles(self, CBVs, CBVs_nearby_vehicles):
        self.CBVs = CBVs
        self.route_scenario.CBVs = CBVs
        self.route_scenario.CBVs_nearby_vehicles = CBVs_nearby_vehicles

    def run_scenario(self):
        self.running = True
        self._init_scenarios()  # generate the background vehicle

    def _init_scenarios(self):
        # spawn route actors for each scenario
        self.route_scenario.initialize_actors()  # generate the background vehicle
    
    def stop_scenario(self):
        self.running = False

    def update_running_status(self, extra_status):
        record, ego_stop, ego_collision, ego_truncated = self.route_scenario.get_running_status(self.running_record)
        record.update(extra_status)  # pass the extra status to the record
        self.running_record.append(record)  # contain every step's record
        # update the status of the scenario manager
        self.running = False if ego_stop else True
        self.ego_collision = True if ego_collision else False
        self.ego_truncated = True if ego_truncated else False

    def get_update(self, timestamp, scenario_action):
        if self._timestamp_last_run < timestamp.elapsed_seconds and self.running:
            self._timestamp_last_run = timestamp.elapsed_seconds
            GameTime.on_carla_tick(timestamp)
            # update the scenario instance receiving the scenario action
            self.scenario_instance.update_behavior(self.CBVs, scenario_action)
            self.route_scenario.activate_background_actors() if len(self.route_scenario.unactivated_actors) > 0 else None
