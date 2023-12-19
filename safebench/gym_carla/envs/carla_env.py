#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：carla_env.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""
import math
import random
import copy

import numpy as np
import pygame
from skimage.transform import resize
import gym
import carla
import time

from safebench.gym_carla.envs.route_planner import RoutePlanner
from safebench.gym_carla.envs.misc import (
    display_to_rgb,
    rgb_to_display_surface,
    get_lane_dis,
    get_pos,
    get_preview_lane_dis,
)
from safebench.agent.agent_utils.explainability_utils import get_masked_viz_3rd_person
from safebench.gym_carla.envs.utils import get_cbv_candidates, get_nearby_vehicles, find_closest_vehicle, \
    get_actor_in_drivable_area, get_ego_min_dis, get_ego_cbv_dis_reward, get_cbv_bv_reward, get_constrain_h, get_min_distance_across_bboxes, \
    reset_ego_cbv_dis, get_cbv_stuck_reward, get_ego_cbv_reward, calculate_abs_velocity
from safebench.scenario.scenario_definition.route_scenario import RouteScenario
from safebench.scenario.scenario_manager.scenario_manager import ScenarioManager
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.agent.agent_utils.visualization import draw_route


class CarlaEnv(gym.Env):
    """ 
        An OpenAI-gym style interface for CARLA simulator. 
    """

    def __init__(self, env_params, birdeye_render=None, display=None, world=None, search_radius=40,
                 safety_network_config=None, agent_state_encoder=None, logger=None):
        assert world is not None, "the world passed into CarlaEnv is None"

        self.config = None
        self.world = world
        self.display = display
        self.logger = logger
        self.birdeye_render = birdeye_render

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0
        self.env_id = None
        self.ego_vehicle = None
        self.constrain_h = None
        self.env_params = env_params
        self.auto_ego = env_params['auto_ego']
        self.enable_sem = env_params['enable_sem']
        self.ego_agent_learnable = env_params['ego_agent_learnable']
        self.mode = env_params['mode']

        self.lidar_sensor = None
        self.camera_sensor = None
        self.sem_sensor = None
        self.lidar_data = None
        self.lidar_height = 2.1

        self.cbv = None
        self.cbv_nearby_vehicles = None
        self.ego_nearby_vehicle = None
        self.old_cbv = None
        self.gps_route = None
        self.route = None
        self.encoded_state = None
        self.collide_with_cbv = False
        self.collide = False
        self.search_radius = search_radius
        self.agent_obs_type = env_params['agent_obs_type']
        self.agent_state_encoder = agent_state_encoder

        # set the safety network's obs type
        if safety_network_config:
            self.safety_network_obs_type = safety_network_config['obs_type']
        else:
            self.safety_network_obs_type = None

        # for Cbv
        self.cbv_select_method = env_params['cbv_selection']

        # scenario manager
        use_scenic = True if env_params['scenario_category'] == 'scenic' else False
        self.scenario_manager = ScenarioManager(env_params, self.logger, use_scenic=use_scenic)

        # for birdeye view and front view visualization
        self.ego_agent_learnable = env_params['ego_agent_learnable']
        self.viz_route = env_params['viz_route']
        self.display_size = env_params['display_size']
        self.obs_range = env_params['obs_range']
        self.d_behind = env_params['d_behind']
        self.disable_lidar = env_params['disable_lidar']

        # for env wrapper
        self.max_past_step = env_params['max_past_step']
        self.max_episode_step = env_params['max_episode_step']
        self.max_waypt = env_params['max_waypt']
        self.lidar_bin = env_params['lidar_bin']
        self.out_lane_thres = env_params['out_lane_thres']
        self.desired_speed = env_params['desired_speed']
        self.acc_max = env_params['continuous_accel_range'][1]
        self.steering_max = env_params['continuous_steer_range'][1]

        # for scenario
        self.ROOT_DIR = env_params['ROOT_DIR']
        self.scenario_category = env_params['scenario_category']
        self.warm_up_steps = env_params['warm_up_steps']

        if self.scenario_category in ['planning', 'scenic']:
            self.obs_size = int(self.obs_range / self.lidar_bin)
        else:
            raise ValueError(f'Unknown scenario category: {self.scenario_category}')

        # action and observation spaces
        self.discrete = env_params['discrete']
        self.discrete_act = [env_params['discrete_acc'], env_params['discrete_steer']]  # acc, steer
        self.n_acc = len(self.discrete_act[0])
        self.n_steer = len(self.discrete_act[1])

    def _create_sensors(self):
        if self.mode == 'eval':
            # lidar sensor
            self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
            self.lidar_bp = CarlaDataProvider._blueprint_library.find('sensor.lidar.ray_cast')
            self.lidar_bp.set_attribute('channels', '16')
            self.lidar_bp.set_attribute('range', '1000')

            # camera sensor
            self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
            self.BGR_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
            # self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))  # for ego view
            self.camera_trans = carla.Transform(carla.Location(x=-4., y=0., z=5.),
                                                carla.Rotation(pitch=-20.0))  # for third-person view
            self.camera_bp = CarlaDataProvider._blueprint_library.find('sensor.camera.rgb')
            # Modify the attributes of the blueprint to set image resolution and field of view.
            self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
            self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
            self.camera_bp.set_attribute('fov', '110')
            # Set the time in seconds between sensor captures
            self.camera_bp.set_attribute('sensor_tick', '0.02')

            # sem camera sensor
            if self.enable_sem:
                self.sem_img = np.zeros((self.obs_size, self.obs_size, 2), dtype=np.uint8)
                self.sem_trans = carla.Transform(carla.Location(x=-4., y=0, z=5.), carla.Rotation(pitch=-20.0))  # for third-person view
                self.sem_bp = CarlaDataProvider._blueprint_library.find('sensor.camera.semantic_segmentation')
                self.sem_bp.set_attribute('image_size_x', str(self.obs_size))
                self.sem_bp.set_attribute('image_size_y', str(self.obs_size))
                self.sem_bp.set_attribute('fov', '110')
                # Set the time in seconds between sensor captures
                self.sem_bp.set_attribute('sensor_tick', '0.02')

    def _create_scenario(self, config, env_id):
        self.logger.log(f">> Loading scenario data id: {config.data_id}")

        # create scenario according to different types
        if self.scenario_category == 'planning':
            scenario = RouteScenario(
                world=self.world,
                config=config,
                ego_id=env_id,
                max_running_step=self.max_episode_step,
                env_params=self.env_params,
                mode=self.mode,
                logger=self.logger
            )
        else:
            raise ValueError(f'Unknown scenario category: {self.scenario_category}')

        # init scenario
        self.ego_vehicle = scenario.ego_vehicle
        self.scenario_manager.load_scenario(scenario)  # The scenario manager only controls the RouteScenario
        self.route = self.scenario_manager.route_scenario.route  # the global route
        self.gps_route = self.scenario_manager.route_scenario.gps_route  # the global gps route

    def _run_scenario(self):
        self.scenario_manager.run_scenario()  # init the background vehicle

    def _global_route_to_waypoints(self):
        waypoints_list = []
        self.carla_map = self.world.get_map()
        for node in self.route:
            loc = node[0].location
            waypoint = self.carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            waypoints_list.append(waypoint)
        return waypoints_list

    def cbv_selection(self):
        cbv_candidates = None
        # get the ego min distance of the next obs (s_t+1)
        if self.safety_network_obs_type:
            self.constrain_h = get_constrain_h(self.ego_vehicle, self.search_radius, self.ego_nearby_vehicle, self.ego_agent_learnable)

        # safety network or agent need "plant" style input or cbv selection is attention based
        if self.agent_state_encoder:
            if not self.safety_network_obs_type:
                cbv_candidates = get_cbv_candidates(self.ego_vehicle, self.search_radius)
            encoded_state, most_relevant_vehicle = self.agent_state_encoder.get_encoded_state(
                self.ego_vehicle, cbv_candidates, self.waypoints, self.red_light_state
            )
            self.encoded_state = encoded_state[:, 0, :].squeeze(0).detach()  # from tensor (1, x, 512) to (1, 512) to (512)
        else:
            most_relevant_vehicle = None
        self.old_cbv = self.cbv  # for BEV visualization
        # filter and sort the background vehicle according to the distance to the ego vehicle in ascending order
        if self.cbv_select_method == 'rule-based':
            self.cbv = find_closest_vehicle(self.ego_vehicle, self.search_radius, cbv_candidates)
        elif self.cbv_select_method == 'attention-based':
            assert most_relevant_vehicle is not None, "attention-based cbv selection will got most relevant vehicle from agent state encoder"
            self.cbv = most_relevant_vehicle

        # get the nearby vehicles around the cbv
        if self.cbv:
            self.cbv_nearby_vehicles = get_nearby_vehicles(self.cbv, self.search_radius)
            if self.birdeye_render:
                # update the cbv for BEV visualization
                if self.old_cbv and self.cbv and self.old_cbv.id != self.cbv.id:  # the cbv has changed, need to remove the old cbv
                    self.birdeye_render.set_cbv(self.cbv, self.cbv.id)  # update the new cbv
                    self.birdeye_render.remove_old_cbv(self.old_cbv.id)  # remove the old cbv if exist
                elif self.old_cbv is not None and self.cbv is None:
                    self.birdeye_render.remove_old_cbv(self.old_cbv.id)  # remove the old cbv if exist
                elif self.old_cbv is None and self.cbv is not None:  # got the initial cbv
                    self.birdeye_render.set_cbv(self.cbv, self.cbv.id)  # add the new cbv
        else:
            self.cbv_nearby_vehicles = None
        self.scenario_manager.update_cbv_nearby_vehicles(self.cbv, self.cbv_nearby_vehicles)

    def reset(self, config, env_id):
        self.config = config
        self.env_id = env_id

        self._create_sensors()
        # create RouteScenario, scenario manager, ego_vehicle etc.
        self._create_scenario(config, env_id)

        # generate the initial background vehicles
        self._run_scenario()
        self._attach_sensor()

        # first update the info in the CarlaDataProvider
        CarlaDataProvider.on_carla_tick()

        # route planner for ego vehicle
        self.global_route_waypoints = self._global_route_to_waypoints()  # the initial route waypoints from the config
        self.routeplanner = RoutePlanner(self.ego_vehicle, self.max_waypt, self.global_route_waypoints)
        self.waypoints, _, _, _, self.red_light_state, self.vehicle_front = self.routeplanner.run_step()

        # find ego nearby vehicles
        self.ego_nearby_vehicle = get_nearby_vehicles(self.ego_vehicle, self.search_radius)

        # set controlled bv
        self.cbv_selection()

        # Get actors polygon list (for visualization)
        if self.birdeye_render:
            self.vehicle_polygons = [self._get_actor_polygons('vehicle.*')]
            self.walker_polygons = [self._get_actor_polygons('walker.*')]

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1

        # applying setting can tick the world and get data from sensors
        # removing this block will cause error: AttributeError: 'NoneType' object has no attribute 'raw_data'
        self.settings = self.world.get_settings()
        self.world.apply_settings(self.settings)

        for _ in range(self.warm_up_steps):
            self.world.tick()
        return self._get_obs(), self._get_info(need_scenario_reward=False, reset=True)

    def _attach_sensor(self):
        if self.mode == 'eval':
            # Add lidar sensor
            if not self.disable_lidar:
                self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego_vehicle)
                self.lidar_sensor.listen(lambda data: get_lidar_data(data))

            def get_lidar_data(data):
                self.lidar_data = data

            # Add camera sensor
            self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego_vehicle)
            self.camera_sensor.listen(lambda data: get_camera_img(data))

            def get_camera_img(data):
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, :3]
                self.BGR_img = copy.deepcopy(array)
                array = array[:, :, ::-1]
                self.camera_img = array

            # Add sem_camera sensor
            if self.enable_sem:
                self.sem_sensor = self.world.spawn_actor(self.sem_bp, self.sem_trans, attach_to=self.ego_vehicle)
                self.sem_sensor.listen(lambda data: get_sem_img(data))

            def get_sem_img(data):
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, 2]  # from PlanT
                self.sem_img = array

    def visualize_ego_route_cbv(self):
        # Visualize the controlled bv
        if self.cbv:
            cbv_transform = CarlaDataProvider.get_transform(self.cbv)
            cbv_begin = carla.Location(x=cbv_transform.location.x, y=cbv_transform.location.y, z=3)
            cbv_angle = math.radians(cbv_transform.rotation.yaw)
            cbv_end = cbv_begin + carla.Location(x=math.cos(cbv_angle), y=math.sin(cbv_angle))
            self.world.debug.draw_arrow(cbv_begin, cbv_end, arrow_size=0.3, color=carla.Color(0, 0, 255, 0),
                                        life_time=0.11)

        # if the ego agent is learnable and need to viz the route, then draw the target waypoints
        if self.ego_agent_learnable and self.viz_route:
            waypoint_route = np.array([[node[0], node[1]] for node in self.waypoints])
            draw_route(self.world, self.ego_vehicle, waypoint_route)

    def step_before_tick(self, ego_action, scenario_action):
        if self.world:
            snapshot = self.world.get_snapshot()
            if snapshot:
                timestamp = snapshot.timestamp

                # update the cbv action
                self.scenario_manager.get_update(timestamp, scenario_action)

                # Calculate acceleration and steering
                if not self.auto_ego:
                    if not self.ego_agent_learnable:
                        # the rule based action
                        throttle = ego_action[0]
                        steer = ego_action[1]
                        brake = ego_action[2]
                        act = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
                    else:
                        # the learnable agent action
                        if self.discrete:
                            acc = self.discrete_act[0][ego_action // self.n_steer]  # 'discrete_acc': [-3.0, 0.0, 3.0]
                            steer = self.discrete_act[1][
                                ego_action % self.n_steer]  # 'discrete_steer': [-0.2, 0.0, 0.2]
                        else:
                            acc = ego_action[0]  # continuous action: acc
                            steer = ego_action[1]  # continuous action: steering

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
                    self.ego_vehicle.apply_control(act)  # apply action of the ego vehicle on the next tick
            else:
                self.logger.log('>> Can not get snapshot!', color='red')
                raise Exception()
        else:
            self.logger.log('>> Please specify a Carla world!', color='red')
            raise Exception()

    def step_after_tick(self):
        if self.birdeye_render:
            # Append actors polygon list
            vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
            self.vehicle_polygons.append(vehicle_poly_dict)
            while len(self.vehicle_polygons) > self.max_past_step:
                self.vehicle_polygons.pop(0)
            walker_poly_dict = self._get_actor_polygons('walker.*')
            self.walker_polygons.append(walker_poly_dict)
            while len(self.walker_polygons) > self.max_past_step:
                self.walker_polygons.pop(0)

        # After tick, update all the actors' velocity map, location map and transform map
        CarlaDataProvider.on_carla_tick()

        # route planner
        # self.waypoints: the waypoints from the waypoints buffer, needed to be followed
        self.waypoints, _, _, _, self.red_light_state, self.vehicle_front, = self.routeplanner.run_step()

        # find ego nearby vehicles
        self.ego_nearby_vehicle = get_nearby_vehicles(self.ego_vehicle, self.search_radius)

        # update the running status and check whether terminate or not
        self.scenario_manager.update_running_status()
        self.collide = self.scenario_manager._collision
        self.collide_with_cbv = self.scenario_manager.collide_with_cbv

        origin_info = self._get_info(need_scenario_reward=True)  # info of old cbv

        # set ego min distance and controlled bv
        self.cbv_selection()

        updated_cbv_info = self._get_info(need_scenario_reward=False)  # info of new cbv

        self.visualize_ego_route_cbv() if self.mode == 'eval' else None  # visualize the controlled bv and the waypoints in clients side after tick

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        return (self._get_obs(), self._get_reward(), self._terminal(), [origin_info, updated_cbv_info])

    def _get_info(self, need_scenario_reward, reset=False):
        info = {}
        # info for scenario agents to take action (scenario obs)
        info.update(self.scenario_manager.route_scenario.update_info())  # add the info of all the actors
        if need_scenario_reward:
            # the total reward for the cbv training
            info['scenario_agent_reward'] = self._get_scenario_reward()
        else:
            reset_ego_cbv_dis(self.ego_vehicle, self.cbv)

        if reset:
            info.update({
                'route_waypoints': self.global_route_waypoints,  # the global route waypoints
                'gps_route': self.gps_route,  # the global gps route
                'route': self.route,  # the global route
            })

        # if train the safety network, need to add the corresponding obs
        if self.safety_network_obs_type:
            # only the safety network is not None, then need to calculate the ego min distance
            info['constrain_h'] = self.constrain_h  # the ego_min_dis with the rest bvs
            if self.safety_network_obs_type == 'plant':
                info['encoded_state'] = self.encoded_state
            elif self.safety_network_obs_type == 'ego_info':
                info['ego_info'] = self.scenario_manager.route_scenario.update_ego_info(ego_nearby_vehicle=self.ego_nearby_vehicle)

        return info

    def _get_actor_polygons(self, filt):
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    # def _get_actor_info(self, filt):
    #     actor_trajectory_dict = {}
    #     actor_acceleration_dict = {}
    #     actor_angular_velocity_dict = {}
    #     actor_velocity_dict = {}
    #
    #     for actor in self.world.get_actors().filter(filt):
    #         actor_trajectory_dict[actor.id] = actor.get_transform()
    #         actor_acceleration_dict[actor.id] = actor.get_acceleration()
    #         actor_angular_velocity_dict[actor.id] = actor.get_angular_velocity()
    #         actor_velocity_dict[actor.id] = actor.get_velocity()
    #     return actor_trajectory_dict, actor_acceleration_dict, actor_angular_velocity_dict, actor_velocity_dict

    def _get_obs(self):
        if self.birdeye_render:
            # set ego information for birdeye_render
            self.birdeye_render.set_hero(self.ego_vehicle, self.ego_vehicle.id)
            self.birdeye_render.vehicle_polygons = self.vehicle_polygons
            self.birdeye_render.walker_polygons = self.walker_polygons
            self.birdeye_render.waypoints = self.waypoints

            # render birdeye image with the birdeye_render
            birdeye_render_types = ['roadmap', 'actors', 'waypoints']
            birdeye_surface = self.birdeye_render.render(birdeye_render_types)
            birdeye_surface = pygame.surfarray.array3d(birdeye_surface)
            center = (int(birdeye_surface.shape[0] / 2), int(birdeye_surface.shape[1] / 2))
            width = height = int(self.display_size / 2)
            birdeye = birdeye_surface[center[0] - width:center[0] + width, center[1] - height:center[1] + height]
            birdeye = display_to_rgb(birdeye, self.obs_size)

            if not self.disable_lidar:
                # get Lidar image
                point_cloud = np.copy(np.frombuffer(self.lidar_data.raw_data, dtype=np.dtype('f4')))
                point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))
                x = point_cloud[:, 0:1]
                y = point_cloud[:, 1:2]
                z = point_cloud[:, 2:3]
                intensity = point_cloud[:, 3:4]
                point_cloud = np.concatenate([y, -x, z], axis=1)
                # Separate the 3D space to bins for point cloud, x and y is set according to self.lidar_bin, and z is set to be two bins.
                y_bins = np.arange(-(self.obs_range - self.d_behind), self.d_behind + self.lidar_bin, self.lidar_bin)
                x_bins = np.arange(-self.obs_range / 2, self.obs_range / 2 + self.lidar_bin, self.lidar_bin)
                z_bins = [-self.lidar_height - 1, -self.lidar_height + 0.25, 1]
                # Get lidar image according to the bins
                lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
                lidar[:, :, 0] = np.array(lidar[:, :, 0] > 0, dtype=np.uint8)
                lidar[:, :, 1] = np.array(lidar[:, :, 1] > 0, dtype=np.uint8)
                wayptimg = birdeye[:, :, 0] < 0  # Equal to a zero matrix
                wayptimg = np.expand_dims(wayptimg, axis=2)
                wayptimg = np.fliplr(np.rot90(wayptimg, 3))
                # Get the final lidar image
                lidar = np.concatenate((lidar, wayptimg), axis=2)
                lidar = np.flip(lidar, axis=1)
                lidar = np.rot90(lidar, 1) * 255

                # display birdeye image
                birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
                self.display.blit(birdeye_surface, (0, self.env_id * self.display_size))

                # display lidar image
                lidar_surface = rgb_to_display_surface(lidar, self.display_size)
                self.display.blit(lidar_surface, (self.display_size, self.env_id * self.display_size))

                # display camera image
                camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
                camera_surface = rgb_to_display_surface(camera, self.display_size)
                self.display.blit(camera_surface, (self.display_size * 2, self.env_id * self.display_size))
            else:
                # display birdeye image
                birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
                self.display.blit(birdeye_surface, (0, self.env_id * self.display_size))

                # display camera image
                camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
                camera_surface = rgb_to_display_surface(camera, self.display_size)
                self.display.blit(camera_surface, (self.display_size, self.env_id * self.display_size))

                # display masked viz 3rd person
                if self.enable_sem:
                    masked_img = get_masked_viz_3rd_person(self.BGR_img, self.sem_img)
                    masked_image = resize(masked_img, (self.obs_size, self.obs_size)) * 255
                    masked_image_surface = rgb_to_display_surface(masked_image, self.display_size)
                    self.display.blit(masked_image_surface, (self.display_size * 2, self.env_id * self.display_size))

        if self.agent_obs_type == 'ego_state':
            # Ego state
            ego_trans = CarlaDataProvider.get_transform(self.ego_vehicle)
            ego_loc = ego_trans.location
            ego_pos = np.array([ego_loc.x, ego_loc.y])
            ego_speed = calculate_abs_velocity(CarlaDataProvider.get_velocity(self.ego_vehicle))  # m/s
            ego_compass = np.deg2rad(ego_trans.rotation.yaw)  # the yaw angle in radius
            ego_state = {
                'gps': ego_pos,
                'speed': ego_speed,
                'compass': ego_compass
            }
            obs = {
                'ego_state': ego_state,
            }
        elif self.agent_obs_type == 'simple_state':
            # default State observation from safebench
            ego_trans = CarlaDataProvider.get_transform(self.ego_vehicle)
            ego_x = ego_trans.location.x
            ego_y = ego_trans.location.y
            ego_yaw = ego_trans.rotation.yaw / 180 * np.pi
            lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
            yaw = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])
            delta_yaw = np.arcsin(np.cross(w, yaw))

            v = CarlaDataProvider.get_velocity(self.ego_vehicle)
            speed = calculate_abs_velocity(v)
            simple_state = np.array([lateral_dis, -delta_yaw, speed, self.vehicle_front])
            obs = {
                'simple_state': simple_state.astype(np.float32),
            }
        elif self.agent_obs_type == 'plant':
            obs = {
                'plant_encoded_state': self.encoded_state.numpy(),
            }
        elif self.agent_obs_type == 'no_obs':
            obs = None
        return obs

    def _get_reward(self):
        """ Calculate the step reward. """
        r_collision = -1 if self.collide else 0

        # reward for steering:
        r_steer = -self.ego_vehicle.get_control().steer ** 2

        # reward for out of lane
        ego_x, ego_y = get_pos(self.ego_vehicle)
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        r_out = -1 if abs(dis) > self.out_lane_thres else 0

        # reward for speed tracking
        v = CarlaDataProvider.get_velocity(self.ego_vehicle)

        # cost for too fast
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)
        r_fast = -1 if lspeed_lon > self.desired_speed else 0

        # cost for lateral acceleration
        r_lat = -abs(self.ego_vehicle.get_control().steer) * lspeed_lon ** 2

        # combine all rewards
        # r = 1 * r_collision + 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat
        # reward from "Interpretable End-to-End Urban Autonomous Driving With Latent Deep Reinforcement Learning"
        r = 50 * r_collision + 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat - 0.1
        return r

    def _get_scenario_reward(self):
        """
            in_drivable_area ~ [0, -1]: whether the cbv are driving on the drivable area
            cbv_min_dis_reward ~ [-1.25, 0]: activate when cbv is too close to the other bvs
            too_fast ~ 0 or -1: if the cbv speed > desired speed, too_fast = -1
            ego_cbv_dis_reward: the delta distance between ego and cbv
        """
        # the min dis from the cbv to the rest bvs
        _, cbv_min_dis_reward = get_cbv_bv_reward(self.cbv, self.search_radius, self.cbv_nearby_vehicles)

        # the min dis from ego to the cbv
        ego_cbv_dis_reward = get_ego_cbv_reward(self.ego_vehicle, self.cbv)

        # the reward design to encourage cbv attack ego
        # ego_cbv_dis_reward = get_ego_cbv_dis_reward(self.ego_vehicle, self.cbv)

        # the reward design to prevent cbv stuck traffic flow
        cbv_stuck_reward = get_cbv_stuck_reward(self.cbv, self.cbv_nearby_vehicles, self.ego_vehicle, self.ego_nearby_vehicle)

        # since the obs don't have road info, so no need to include in drivable area info
        # in_drivable_area = get_actor_in_drivable_area(self.cbv) if self.cbv else 0

        # ego collision reward, depending on the ego min distance
        ego_cbv_collide_reward = 1 if self.collide_with_cbv else 0

        # final scenario agent rewards
        scenario_agent_reward = 20 * cbv_min_dis_reward + 0.5 * ego_cbv_dis_reward + 500 * ego_cbv_collide_reward + 5 * cbv_stuck_reward

        return scenario_agent_reward

    def _get_cost(self):
        # cost for collision
        r_collision = 0
        if self.collide:
            r_collision = -1
        return r_collision

    def _terminal(self):
        return not self.scenario_manager._running

    def _remove_sensor(self):
        if self.lidar_sensor is not None:
            self.lidar_sensor.stop()
            self.lidar_sensor.destroy()
            self.lidar_sensor = None
        if self.camera_sensor is not None:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
            self.camera_sensor = None
        if self.sem_sensor is not None:
            self.sem_sensor.stop()
            self.sem_sensor.destroy()
            self.sem_sensor = None

    def _remove_ego(self):
        if self.ego_vehicle is not None and CarlaDataProvider.actor_id_exists(self.ego_vehicle.id):
            CarlaDataProvider.remove_actor_by_id(self.ego_vehicle.id)
        self.ego_vehicle = None

    def _reset_variables(self):
        self.old_cbv = None
        self.cbv = None
        self.ego_nearby_vehicle = None
        self.cbv_nearby_vehicles = None
        self.old_cbv = None
        self.gps_route = None
        self.route = None
        self.encoded_state = None
        self.constrain_h = None
        self.global_route_waypoints = None
        self.waypoints = None
        self.collide_with_cbv = False
        self.collide = False

    def clean_up(self):
        # remove temp variables
        self._reset_variables()

        # remove the render sensor only when evaluating
        self._remove_sensor()

        # destroy criterion sensors on the ego vehicle
        self.scenario_manager.clean_up()

        # remove the ego vehicle after removing all the sensors
        self._remove_ego()

