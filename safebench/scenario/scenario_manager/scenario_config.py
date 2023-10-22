#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：scenario_config.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
'''

import carla


class ScenarioConfig(object):
    """
        Configuration of parsed scenario
    """

    auto_ego = False
    num_scenario = None
    route_region = ''
    data_id = 0
    scenario_folder = None
    scenario_id = 0
    route_id = 0
    risk_level = 0
    parameters = None

    town = ''
    name = ''
    weather = None
    scenario_file = None
    initial_transform = None
    initial_pose = None
    trajectory = None
    texture_dir = None


class RouteScenarioConfig(object):
    """
        configuration of a RouteScenario
    """
    other_actors = []
    trigger_points = []
    route_var_name = None
    subtype = None
    parameters = None
    weather = carla.WeatherParameters()
    num_scenario = None
    friction = None


