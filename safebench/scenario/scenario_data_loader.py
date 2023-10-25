#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：scenario_data_loader.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""

import numpy as np
from copy import deepcopy
from safebench.util.torch_util import set_seed
from safebench.scenario.tools.route_manipulation import interpolate_trajectory


def calculate_interpolate_trajectory(config, world):
    # get route
    origin_waypoints_loc = []
    for loc in config.trajectory:
        origin_waypoints_loc.append(loc)
    _, route = interpolate_trajectory(world, origin_waypoints_loc, 5.0)

    # get [x, y] along the route
    waypoint_xy = []
    for transform_tuple in route:
        waypoint_xy.append([transform_tuple[0].location.x, transform_tuple[0].location.y])

    return waypoint_xy


def check_route_overlap(current_routes, route, distance_threshold=10):
    overlap = False
    for current_route in current_routes:
        for current_waypoint in current_route:
            for waypoint in route:
                distance = np.linalg.norm([current_waypoint[0] - waypoint[0], current_waypoint[1] - waypoint[1]])
                if distance < distance_threshold:
                    overlap = True
                    return overlap

    return overlap


class ScenarioDataLoader:
    def __init__(self, config_lists, num_scenario, town, world):
        self.num_scenario = num_scenario
        self.config_lists = config_lists  # can be changed during training
        self.constant_config_lists = config_lists  # unchangeable
        self.town = town.lower()
        self.world = world
        self.routes = []

        # If using CARLA maps, manually check overlaps
        if 'safebench' not in self.town:
            for config in config_lists:
                self.routes.append(calculate_interpolate_trajectory(config, world))

        self.num_total_scenario = len(config_lists)  # the number of scenarios in one town map
        self.reset_idx_counter()

    def reset_idx_counter(self):
        self.num_total_scenario = len(self.config_lists)  # update num_scenarios after sampling
        if self.num_total_scenario == 0:  # during training, if no more config_lists, need to start over
            # create the new config_lists
            self.config_lists = self.constant_config_lists
            # update the num_total_scenario
            self.num_total_scenario = len(self.config_lists)
            # create the new routes
            if 'safebench' not in self.town:
                self.routes = []
                for config in self.config_lists:
                    self.routes.append(calculate_interpolate_trajectory(config, self.world))
        self.scenario_idx = list(range(self.num_total_scenario))  # both for training and evaluating

    def _select_non_overlap_idx_safebench(self, remaining_ids, sample_num):
        selected_idx = []
        current_regions = []
        for s_i in remaining_ids:
            if self.config_lists[s_i].route_region not in current_regions:
                selected_idx.append(s_i)
                if self.config_lists[s_i].route_region != "random":
                    current_regions.append(self.config_lists[s_i].route_region)
            if len(selected_idx) >= sample_num:
                break
        return selected_idx

    def _select_non_overlap_idx_carla(self, remaining_ids, sample_num):
        selected_idx = []
        selected_routes = []
        for s_i in remaining_ids:
            # the selected sample_num of routes should not overlap with each other
            if not check_route_overlap(selected_routes, self.routes[s_i]):
                selected_idx.append(s_i)
                selected_routes.append(self.routes[s_i])
            if len(selected_idx) >= sample_num:
                break
        return selected_idx

    def _select_non_overlap_idx(self, remaining_ids, sample_num):
        if 'safebench' in self.town:
            # If using SafeBench map, check overlap based on regions
            return self._select_non_overlap_idx_safebench(remaining_ids, sample_num)
        else:
            # If using CARLA maps, manually check overlaps
            return self._select_non_overlap_idx_carla(remaining_ids, sample_num)

    def __len__(self):
        return len(self.scenario_idx)

    def set_mode(self, mode):
        self.mode = mode

    def sampler(self):
        # sometimes the length of list is smaller than num_scenario
        sample_num = np.min([self.num_scenario, len(self.scenario_idx)])
        # select scenarios
        # selected_idx = np.random.choice(self.scenario_idx, size=sample_num, replace=False)
        selected_idx = self._select_non_overlap_idx(self.scenario_idx, sample_num)
        selected_scenario = []
        if self.mode == "train":
            for s_i in selected_idx:
                selected_scenario.append(self.config_lists[s_i])
                # self.scenario_idx.remove(s_i)  # for evaluation

            # removing the selected scenario for the next sampling during training
            self.config_lists = [self.config_lists[i] for i in range(len(self.config_lists)) if i not in selected_idx]
            # removing the selected routes for the next sampling during training
            self.routes = [self.routes[i] for i in range(len(self.routes)) if i not in selected_idx]
        elif self.mode == "eval":
            for s_i in selected_idx:
                selected_scenario.append(self.constant_config_lists[s_i])  # need to be a stable config list
                self.scenario_idx.remove(s_i)  # use self.scenario_idx to represent the remaining_idx

        assert len(selected_scenario) <= self.num_scenario, f"number of scenarios is larger than {self.num_scenario}"
        return selected_scenario, len(selected_scenario)


class ScenicDataLoader:
    def __init__(self, scenic, config, num_scenario, seed = 0):
        self.num_scenario = num_scenario
        self.config = config
        self.behavior = config.behavior
        self.scene_index = config.scene_index
        self.select_num = config.select_num
        self.num_total_scenario = len(self.scene_index)
        self.reset_idx_counter()
        self.seed = seed
        self.generate_scene(scenic)
        
    def generate_scene(self, scenic):
        set_seed(self.seed)
        self.scene = []
        while len(self.scene) < self.config.sample_num:
            scene, _ = scenic.generateScene()
            if scenic.setSimulation(scene):
                self.scene.append(scene)
                scenic.endSimulation()
            
    def reset_idx_counter(self):
        self.scenario_idx = self.scene_index

    def __len__(self):
        return len(self.scenario_idx)

    def sampler(self):
        ## no need to be random for scenic loading file ###
        selected_scenario = []
        idx = self.scenario_idx.pop(0)
        new_config = deepcopy(self.config)
        new_config.scene = self.scene[idx]
        new_config.data_id = idx
        try:
            new_config.trajectory = self.scenicToCarlaLocation(new_config.scene.params['Trajectory'])
        except:
            new_config.trajectory = []
        selected_scenario.append(new_config)
        assert len(selected_scenario) <= self.num_scenario, f"number of scenarios is larger than {self.num_scenario}"
        return selected_scenario, len(selected_scenario)

    def scenicToCarlaLocation(self, points):
        waypoints = []
        for point in points:
            location = carla.Location(point[0], -point[1], 0.0)
            waypoint = CarlaDataProvider.get_map().get_waypoint(location)
            location.z = waypoint.transform.location.z + 0.5
            waypoints.append(location)
        return waypoints