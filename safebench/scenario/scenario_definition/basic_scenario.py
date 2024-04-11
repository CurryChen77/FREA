#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：basic_scenario.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""
import carla

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider


class BasicScenario(object):
    """
        Base class for user-defined scenario
    """
    def __init__(self, name, config, world):
        self.world = world
        self.name = name
        self.config = config

        self.ego_vehicles = None

        if CarlaDataProvider.is_sync_mode():
            world.tick()
        else:
            world.wait_for_tick()

    def create_behavior(self, scenario_init_action):
        """
            This method defines the initial behavior of the scenario
        """
        raise NotImplementedError(
            "This function is re-implemented by all scenarios. If this error becomes visible the class hierarchy is somehow broken")

    def update_behavior(self, scenario_action):
        """
            This method defines how to update the behavior of the actors in scenario in each step.
        """
        raise NotImplementedError(
                "This function is re-implemented by all scenarios. If this error becomes visible the class hierarchy is somehow broken")

    def initialize_actors(self):
        """
            This method defines how to initialize the actors in scenario.
        """
        raise NotImplementedError(
                "This function is re-implemented by all scenarios. If this error becomes visible the class hierarchy is somehow broken")

    def check_stop_condition(self):
        """
            This method defines the stop condition of the scenario.
        """
        raise NotImplementedError(
            "This function is re-implemented by all scenarios. If this error becomes visible the class hierarchy is somehow broken")

    def clean_up(self):
        pass


