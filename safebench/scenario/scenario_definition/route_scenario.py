#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：route_scenario.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""
import time
import traceback

import numpy as np
import carla

from safebench.gym_carla.envs.utils import get_locations_nearby_spawn_points, calculate_abs_velocity
from safebench.scenario.scenario_manager.timer import GameTime
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider

from safebench.scenario.tools.route_manipulation import interpolate_trajectory
from safebench.scenario.tools.scenario_utils import (
    get_valid_spawn_points,
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
    StuckDetectorTest,
    RunningRedLightTest,
    RunningStopTest,
)


class RouteScenario():
    """
        Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
        along which several smaller scenarios are triggered
    """

    def __init__(self, world, config, ego_id, max_running_step, env_params, mode, logger):
        self.world = world
        self.logger = logger
        self.config = config
        self.ego_id = ego_id
        self.max_running_step = max_running_step
        self.mode = mode
        self.timeout = 60
        self.ego_max_driven_distance = 200
        self.traffic_intensity = config.traffic_intensity
        self.search_radius = config.search_radius

        # create the route and ego's position (the start point of the route)
        self.route, self.ego_vehicle, self.gps_route = self._update_route_and_ego(timeout=self.timeout)
        self.background_actors = []
        self.cbv = None
        self.cbv_nearby_vehicles = None
        self.criteria = self._create_criteria()
        self.scenario_instance = AdvBehaviorSingle(self.world, self.ego_vehicle, env_params)  # create the scenario instance

    def _update_route_and_ego(self, timeout=None):
        ego_vehicle = None
        route = None
        gps_route = None
        scenario_id = self.config.scenario_id
        if scenario_id == 0:
            vehicle_spawn_points = get_valid_spawn_points(self.world)
            for random_transform in vehicle_spawn_points:
                gps_route, route = interpolate_trajectory(self.world, [random_transform])
                ego_vehicle = self._spawn_ego_vehicle(route[0][0], self.config.auto_ego)
                if ego_vehicle is not None:
                    break
        else:
            gps_route, route = interpolate_trajectory(self.world, self.config.trajectory)
            ego_vehicle = self._spawn_ego_vehicle(route[0][0], self.config.auto_ego)

        # TODO: ego route will be overwritten by other scenarios
        CarlaDataProvider.set_ego_vehicle_route(ego_vehicle, convert_transform_to_location(route))
        CarlaDataProvider.set_scenario_config(self.config)

        # Timeout of scenario in seconds
        self.timeout = self._estimate_route_timeout(route) if timeout is None else timeout
        return route, ego_vehicle, gps_route

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

    def get_location_nearby_spawn_points(self):
        start_location = self.route[0][0].location
        middle_location = self.route[len(self.route)//2][0].location
        end_location = self.route[-1][0].location
        locations_list = [start_location, middle_location, end_location]
        radius_list = [10, 40, 40]
        closest_dis = 7

        spawn_points = get_locations_nearby_spawn_points(
            locations_list, radius_list, closest_dis, self.traffic_intensity
        )
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
            'ego_velocity': calculate_abs_velocity(CarlaDataProvider.get_velocity(self.ego_vehicle)),
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
        collide_with_cbv = False
        # collision with other objects
        if running_status['collision'][0] == Status.FAILURE:
            stop = True
            collision = True
            if running_status['collision'][1] is not None and self.cbv is not None and running_status['collision'][1] == self.cbv.id:
                collide_with_cbv = True
                self.logger.log(f'>> Scenario stops due to collision with cbv', color='yellow')
            else:
                self.logger.log(f'>> Scenario stops due to collision with normal object', color='yellow')

        # out of the road detection
        if running_status['off_road'] == Status.FAILURE:
            stop = True
            self.logger.log('>> Scenario stops due to off road', color='yellow')

        # stuck
        if self.mode != 'eval' and running_status['stuck'] == Status.FAILURE:
            stop = True
            self.logger.log('>> Scenario stops due to stuck', color='yellow')

        # route completed
        if running_status['route_complete'] == 100:
            stop = True
            self.logger.log('>> Scenario stops due to route completion', color='yellow')

        # stop at max step
        if len(running_record) >= self.max_running_step: 
            stop = True
            self.logger.log('>> Scenario stops due to max steps', color='yellow')

        if running_status['current_game_time'] >= self.timeout:
            stop = True
            self.logger.log('>> Scenario stops due to timeout', color='yellow')

        return running_status, stop, collision, collide_with_cbv

    def _create_criteria(self):
        criteria = {}
        route = convert_transform_to_location(self.route)

        # the criteria needed both in training and evaluating
        criteria['route_complete'] = RouteCompletionTest(self.ego_vehicle, route=route)
        criteria['off_road'] = OffRoadTest(actor=self.ego_vehicle, optional=True)
        criteria['collision'] = CollisionTest(actor=self.ego_vehicle, terminate_on_failure=True)  # need sensor
        if self.mode == 'eval':
            # extra criteria for evaluating
            criteria['driven_distance'] = DrivenDistanceTest(actor=self.ego_vehicle, distance_success=1e4, distance_acceptable=1e4, optional=True)
            criteria['distance_to_route'] = InRouteTest(self.ego_vehicle, route=route, offroad_max=30)
            criteria['lane_invasion'] = KeepLaneTest(actor=self.ego_vehicle, optional=True)  # need sensor
        else:
            criteria['stuck'] = StuckDetectorTest(actor=self.ego_vehicle, len_thresh=100, speed_thresh=0.1, terminate_on_failure=True)

        return criteria

    @staticmethod
    def _get_actor_state(actor):
        actor_trans = CarlaDataProvider.get_transform(actor)
        actor_x = actor_trans.location.x
        actor_y = actor_trans.location.y
        actor_yaw = round(actor_trans.rotation.yaw / 180 * np.pi, 3)
        # yaw = np.array([np.cos(actor_yaw), np.sin(actor_yaw)])
        velocity = CarlaDataProvider.get_velocity(actor)
        # acc = actor.get_acceleration()
        # [actor_x, actor_y, actor_yaw, yaw[0], yaw[1], velocity.x, velocity.y, acc.x, acc.y]
        return [actor_x, actor_y, actor_yaw, velocity.x, velocity.y]

    def update_info(self, desired_nearby_vehicle=3):
        '''
            scenario agent state:
            first row is ego's relative state (x, y, yaw, vx, vy)
            rest row are other bv's relative state (x, y, yaw, vx, vy)
        '''
        if self.cbv:  # the cbv is not None
            # absolute state
            cbv_state = np.array(self._get_actor_state(self.cbv))
            # relative state
            ego_state = np.array(self._get_actor_state(self.ego_vehicle)) - cbv_state

            actor_info = [ego_state]  # the first info belongs to the ego vehicle
            for actor in self.cbv_nearby_vehicles:
                if actor.id == self.ego_vehicle.id:
                    continue  # except the ego actor
                elif len(actor_info) < desired_nearby_vehicle:
                    actor_state = np.array(self._get_actor_state(actor)) - cbv_state
                    actor_info.append(actor_state)  # add the info of the other actor to the list
                else:
                    # avoiding too many nearby vehicles
                    break
            while len(actor_info) < desired_nearby_vehicle:  # if no enough nearby vehicles, padding with 0
                actor_info.append([0] * len(cbv_state))

            actor_info = np.array(actor_info)
        else:
            actor_info = None
        return {
            'scenario_obs': actor_info  # the controlled bv on the first line, while the rest bvs are sorted in ascending order
        }

    def update_ego_info(self, ego_nearby_vehicles, desired_nearby_vehicle=3):
        '''
            safety network input state:
            all the rows are other bv's relative state
        '''
        # absolute ego state
        ego_state = np.array(self._get_actor_state(self.ego_vehicle))
        # relative ego state
        ego_info = []
        for actor in enumerate(ego_nearby_vehicles):
            if len(ego_info) < desired_nearby_vehicle:
                actor_state = np.array(self._get_actor_state(actor)) - ego_state
                ego_info.append(actor_state)  # all the row contain meaningful vehicle around ego vehicle
            else:
                break
        while len(ego_info) < desired_nearby_vehicle:  # if no enough nearby vehicles, padding with 0
            ego_info.append([0] * len(ego_state))

        ego_info = np.array(ego_info)
        # get the info of the ego vehicle and the other actors
        return ego_info

    def clean_up(self):
        # stop criterion and destroy sensors
        for _, criterion in self.criteria.items():
            criterion.terminate()
        time.sleep(0.1)

        self.scenario_instance.clean_up()  # nothing need to clean

        # clean background vehicle (the vehicle will be destroyed in CarlaDataProvider)
        self.background_actors = []


