#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：route_scenario.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""
import traceback

import numpy as np
import carla

from safebench.util.run_util import class_from_path
from safebench.scenario.scenario_manager.timer import GameTime
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_manager.scenario_config import RouteScenarioConfig
from safebench.scenario.tools.route_parser import RouteParser
from safebench.scenario.tools.route_manipulation import interpolate_trajectory
from safebench.scenario.tools.scenario_utils import (
    get_valid_spawn_points, 
    convert_json_to_transform, 
    convert_json_to_actor, 
    convert_transform_to_location
)
from safebench.scenario.scenario_definition.adv_behavior_scenario import AdvBehaviorSingle
from safebench.scenario.scenario_definition.atomic_criteria import (
    Status,
    CollisionTest,
    DrivenDistanceTest,
    AverageVelocityTest,
    OffRoadTest,
    KeepLaneTest,
    InRouteTest,
    RouteCompletionTest,
    RunningRedLightTest,
    RunningStopTest,
)


class RouteScenario():
    """
        Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
        along which several smaller scenarios are triggered
    """

    def __init__(self, world, config, ego_id, max_running_step, env_params, logger):
        self.world = world
        self.logger = logger
        self.config = config
        self.ego_id = ego_id
        self.max_running_step = max_running_step
        self.timeout = 60
        self.ego_max_driven_distance = 200
        self.traffic_intensity = config.traffic_intensity
        self.search_radius = config.search_radius

        # create the route and ego's position (the start point of the route)
        self.route, self.ego_vehicle = self._update_route_and_ego(timeout=self.timeout)
        # self.route, self.ego_vehicle, scenario_definitions = self._update_route_and_ego(timeout=self.timeout)
        self.background_actors = []
        self.controlled_bv = None
        self.controlled_bv_nearby_vehicles = None
        # scenario_definitions contains the possible scenarios along the pre-defined route
        # self.list_scenarios = self._build_scenario_instances(scenario_definitions)  # remove the scenario_definitions
        self.criteria = self._create_criteria()
        self.scenario_instance = AdvBehaviorSingle(self.world, self.ego_vehicle, env_params)  # create the scenario instance

    def _update_route_and_ego(self, timeout=None):
        # # transform the scenario file into a dictionary
        # if self.config.scenario_file is not None:
        #     world_annotations = RouteParser.parse_annotations_file(self.config.scenario_file)
        # else:
        #     world_annotations = self.config.scenario_config

        # prepare route's trajectory (interpolate and add the GPS route)
        ego_vehicle = None
        route = None
        scenario_id = self.config.scenario_id
        if scenario_id == 0:
            vehicle_spawn_points = get_valid_spawn_points(self.world)
            for random_transform in vehicle_spawn_points:
                _, route = interpolate_trajectory(self.world, [random_transform])
                ego_vehicle = self._spawn_ego_vehicle(route[0][0], self.config.auto_ego)
                if ego_vehicle is not None:
                    break
        else:
            _, route = interpolate_trajectory(self.world, self.config.trajectory)
            ego_vehicle = self._spawn_ego_vehicle(route[0][0], self.config.auto_ego)

        # TODO: ego route will be overwritten by other scenarios
        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(route))
        CarlaDataProvider.set_scenario_config(self.config)

        # Timeout of scenario in seconds
        self.timeout = self._estimate_route_timeout(route) if timeout is None else timeout
        return route, ego_vehicle

    def _estimate_route_timeout(self, route):
        route_length = 0.0  # in meters
        min_length = 100.0
        SECONDS_GIVEN_PER_METERS = 1

        if len(route) == 1:
            return int(SECONDS_GIVEN_PER_METERS * min_length)

        prev_point = route[0][0]
        for current_point, _ in route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point
        return int(SECONDS_GIVEN_PER_METERS * route_length)

    def _spawn_ego_vehicle(self, elevate_transform, autopilot=False):
        role_name = 'ego_vehicle' + str(self.ego_id)

        success = False
        ego_vehicle = None
        while not success:
            try:
                ego_vehicle = CarlaDataProvider.request_new_actor(
                    'vehicle.lincoln.mkz_2017',
                    elevate_transform,
                    rolename=role_name, 
                    autopilot=autopilot
                )
                ego_vehicle.set_autopilot(autopilot, CarlaDataProvider.get_traffic_manager_port())
                success = True
            except RuntimeError:
                print("WARNING: Failed to spawn the ego vehicle, try to modify the z position of the spawn point")
                elevate_transform.location.z += 0.1
        return ego_vehicle

    # def _build_scenario_instances(self, scenario_definitions):
    #     """
    #         Based on the parsed route and possible scenarios, build all the scenario classes.
    #     """
    #     scenario_instance_list = []
    #     for _, definition in enumerate(scenario_definitions):
    #         # get the class of the scenario
    #         scenario_path = [
    #             'safebench.scenario.scenario_definition',
    #             self.config.scenario_folder,
    #             definition['name'],
    #         ]
    #         scenario_class = class_from_path('.'.join(scenario_path))
    #
    #         # create the other actors that are going to appear
    #         if definition['other_actors'] is not None:
    #             list_of_actor_conf_instances = self._get_actors_instances(definition['other_actors'])
    #         else:
    #             list_of_actor_conf_instances = []
    #
    #         # create an actor configuration for the ego-vehicle trigger position
    #         egoactor_trigger_position = convert_json_to_transform(definition['trigger_position'])
    #         route_config = RouteScenarioConfig()
    #         route_config.other_actors = list_of_actor_conf_instances
    #         route_config.trigger_points = [egoactor_trigger_position]
    #         route_config.parameters = self.config.parameters
    #         route_config.num_scenario = self.config.num_scenario
    #         if self.config.weather is not None:
    #             route_config.weather = self.config.weather
    #
    #         try:
    #             scenario_instance = scenario_class(self.world, self.ego_vehicle, route_config, timeout=self.timeout)
    #         except Exception as e:
    #             traceback.print_exc()
    #             print("Skipping scenario '{}' due to setup error: {}".format(definition['name'], e))
    #             continue
    #
    #         scenario_instance_list.append(scenario_instance)
    #     return scenario_instance_list

    # def _get_actors_instances(self, list_of_antagonist_actors):
    #     def get_actors_from_list(list_of_actor_def):
    #         # receives a list of actor definitions and creates an actual list of ActorConfigurationObjects
    #         sublist_of_actors = []
    #         for actor_def in list_of_actor_def:
    #             sublist_of_actors.append(convert_json_to_actor(actor_def))
    #         return sublist_of_actors
    #
    #     list_of_actors = []
    #     if 'front' in list_of_antagonist_actors:
    #         list_of_actors += get_actors_from_list(list_of_antagonist_actors['front'])
    #     if 'left' in list_of_antagonist_actors:
    #         list_of_actors += get_actors_from_list(list_of_antagonist_actors['left'])
    #     if 'right' in list_of_antagonist_actors:
    #         list_of_actors += get_actors_from_list(list_of_antagonist_actors['right'])
    #     return list_of_actors

    def get_location_nearby_spawn_points(self):
        start_location = self.route[0][0].location
        middle_location = self.route[len(self.route)//2][0].location
        end_location = self.route[-1][0].location
        start = CarlaDataProvider.get_location_nearby_spawn_points(start_location, radius=20, closest_dis=7, intensity=self.traffic_intensity[0])  # route start point
        middle = CarlaDataProvider.get_location_nearby_spawn_points(middle_location, radius=40, intensity=self.traffic_intensity[1])  # route middle point
        end = CarlaDataProvider.get_location_nearby_spawn_points(end_location, radius=30, intensity=self.traffic_intensity[2])  # route end point
        spawn_points = list(set(start + middle + end))  # filter the overlapping
        amount = len(spawn_points)
        return amount, spawn_points

    def initialize_actors(self):
        amount, spawn_points = self.get_location_nearby_spawn_points()
        new_actors = CarlaDataProvider.request_new_batch_actors(
            model='vehicle.*',
            amount=amount,
            spawn_points=spawn_points,
            autopilot=True, 
            random_location=False,
            rolename='background'
        )
        if new_actors is None:
            raise Exception("Error: Unable to add the background activity, all spawn points were occupied")
        self.logger.log(f'>> successfully spawning {len(new_actors)} Autopilot vehicles', color='green')
        for _actor in new_actors:
            self.background_actors.append(_actor)

    def get_running_status(self, running_record):
        running_status = {
            'ego_velocity': CarlaDataProvider.get_velocity(self.ego_vehicle),
            'ego_acceleration_x': self.ego_vehicle.get_acceleration().x,
            'ego_acceleration_y': self.ego_vehicle.get_acceleration().y,
            'ego_acceleration_z': self.ego_vehicle.get_acceleration().z,
            'ego_x': CarlaDataProvider.get_transform(self.ego_vehicle).location.x,
            'ego_y': CarlaDataProvider.get_transform(self.ego_vehicle).location.y,
            'ego_z': CarlaDataProvider.get_transform(self.ego_vehicle).location.z,
            'ego_roll': CarlaDataProvider.get_transform(self.ego_vehicle).rotation.roll,
            'ego_pitch': CarlaDataProvider.get_transform(self.ego_vehicle).rotation.pitch,
            'ego_yaw': CarlaDataProvider.get_transform(self.ego_vehicle).rotation.yaw,
            'current_game_time': GameTime.get_time()
        }

        for criterion_name, criterion in self.criteria.items():
            running_status[criterion_name] = criterion.update()

        stop = False
        collision = False
        # collision with other objects
        if running_status['collision'] == Status.FAILURE:
            stop = True
            collision = True
            self.logger.log(f'>> Scenario stops due to collision', color='yellow')

        # out of the road detection
        if running_status['off_road'] == Status.FAILURE:
            stop = True
            self.logger.log('>> Scenario stops due to off road', color='yellow')

        # only check when evaluating
        if self.config.scenario_id != 0:  
            # route completed
            if running_status['route_complete'] == 100:
                stop = True
                self.logger.log('>> Scenario stops due to route completion', color='yellow')

        # stop at max step
        if len(running_record) >= self.max_running_step: 
            stop = True
            self.logger.log('>> Scenario stops due to max steps', color='yellow')

        # only check when evaluating
        if self.config.scenario_id != 0:
            if running_status['driven_distance'] >= self.ego_max_driven_distance:
                stop = True
                self.logger.log('>> Scenario stops due to max driven distance', color='yellow')
        if running_status['current_game_time'] >= self.timeout:
            stop = True
            self.logger.log('>> Scenario stops due to timeout', color='yellow')

        # for scenario in self.list_scenarios:
        #     # only check when evaluating
        #     if self.config.scenario_id != 0:
        #         if running_status['driven_distance'] >= scenario.ego_max_driven_distance:
        #             stop = True
        #             self.logger.log('>> Scenario stops due to max driven distance', color='yellow')
        #             break
        #     if running_status['current_game_time'] >= scenario.timeout:
        #         stop = True
        #         self.logger.log('>> Scenario stops due to timeout', color='yellow')
        #         break

        return running_status, stop, collision

    def _create_criteria(self):
        criteria = {}
        route = convert_transform_to_location(self.route)

        criteria['driven_distance'] = DrivenDistanceTest(actor=self.ego_vehicle, distance_success=1e4, distance_acceptable=1e4, optional=True)
        criteria['average_velocity'] = AverageVelocityTest(actor=self.ego_vehicle, avg_velocity_success=1e4, avg_velocity_acceptable=1e4, optional=True)
        criteria['lane_invasion'] = KeepLaneTest(actor=self.ego_vehicle, optional=True)
        criteria['off_road'] = OffRoadTest(actor=self.ego_vehicle, optional=True)
        criteria['collision'] = CollisionTest(actor=self.ego_vehicle, terminate_on_failure=True)
        criteria['run_red_light'] = RunningRedLightTest(actor=self.ego_vehicle)
        criteria['run_stop'] = RunningStopTest(actor=self.ego_vehicle)
        if self.config.scenario_id != 0:  # only check when evaluating
            criteria['distance_to_route'] = InRouteTest(self.ego_vehicle, route=route, offroad_max=30)
            criteria['route_complete'] = RouteCompletionTest(self.ego_vehicle, route=route)
        return criteria

    @staticmethod
    def _get_actor_state(actor):
        actor_trans = CarlaDataProvider.get_transform_after_tick(actor)
        actor_x = actor_trans.location.x
        actor_y = actor_trans.location.y
        actor_yaw = actor_trans.rotation.yaw / 180 * np.pi
        yaw = np.array([np.cos(actor_yaw), np.sin(actor_yaw)])
        velocity = actor.get_velocity()
        acc = actor.get_acceleration()
        return [actor_x, actor_y, actor_yaw, yaw[0], yaw[1], velocity.x, velocity.y, acc.x, acc.y]

    def update_info(self, desired_nearby_vehicle=3):
        if self.controlled_bv:  # the controlled bv is not None
            controlled_bv_state = self._get_actor_state(self.controlled_bv)
            actor_info = [controlled_bv_state]  # the first info belongs to the ego vehicle
            for i, actor in enumerate(self.controlled_bv_nearby_vehicles):
                if i < desired_nearby_vehicle:
                    actor_info.append(self._get_actor_state(actor))  # add the info of the other actor to the list
                else:
                    # avoiding too many nearby vehicles
                    break
            while len(actor_info)-1 < desired_nearby_vehicle:  # if no enough nearby vehicles, padding with 0
                actor_info.append([0] * len(controlled_bv_state))

                # for s_i in self.list_scenarios:
                #     for a_i in s_i.other_actors:  # The order of the other actors follows the order in the .json file
                #         actor_state = self._get_actor_state(a_i)
                #         actor_info.append(actor_state)  # add the info of the other actor to the list
            actor_info = np.array(actor_info)
        else:
            actor_info = np.zeros((desired_nearby_vehicle+1, 9))  # need to have the same size like normal actor info
        return {
            'actor_info': actor_info  # the controlled bv on the first line, while the rest bvs are sorted in ascending order
        }

    def clean_up(self):
        # stop criterion and destroy sensors
        for _, criterion in self.criteria.items():
            criterion.terminate()

        # # each scenario remove its own actors
        # for scenario in self.list_scenarios:
        #     scenario.clean_up()
        self.scenario_instance.clean_up()

        # remove background vehicles
        for s_i in range(len(self.background_actors)):
            if self.background_actors[s_i].type_id.startswith('vehicle'):
                self.background_actors[s_i].set_autopilot(enabled=False)
            if CarlaDataProvider.actor_id_exists(self.background_actors[s_i].id):
                CarlaDataProvider.remove_actor_by_id(self.background_actors[s_i].id)
        self.logger.log(f'>> cleaning {len(self.background_actors)} vehicles', color='yellow')
        self.background_actors = []
