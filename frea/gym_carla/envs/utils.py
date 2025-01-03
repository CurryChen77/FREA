#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：utils.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/12/5
"""

import carla
import math

import torch
from rdp import rdp

import numpy as np
from distance3d import gjk, colliders

from frea.gym_carla.envs.misc import is_within_distance_ahead
from frea.scenario.tools.scenario_utils import compute_box2origin
from frea.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from frea.util.torch_util import CUDA, CPU


def linear_map(value, original_range, desired_range):
    """Linear map of value with original range to desired range."""
    return desired_range[0] + (value - original_range[0]) * (desired_range[1] - desired_range[0]) / (original_range[1] - original_range[0])


def get_actor_off_road(actor):
    current_location = CarlaDataProvider.get_location(actor)

    # Get the waypoint at the current location to see if the actor is offroad
    drive_waypoint = CarlaDataProvider.get_map().get_waypoint(current_location, project_to_road=False)
    park_waypoint = CarlaDataProvider.get_map().get_waypoint(current_location, project_to_road=False, lane_type=carla.LaneType.Parking)
    if drive_waypoint or park_waypoint:
        off_road = False
    else:
        off_road = True
    return off_road


def get_closest_CBV_flag(ego_nearby_vehicles, CBVs):
    CBVs_closest_flag = {}
    closest_BV_id = None
    if len(ego_nearby_vehicles) >= 1:
        closest_BV_id = ego_nearby_vehicles[0].id
    for CBV_id, CBV in CBVs.items():
        CBVs_closest_flag[CBV_id] = (CBV_id == closest_BV_id)
    return CBVs_closest_flag


def process_ego_action(ego_action, acc_range, steering_range):
    # the learnable agent action
    throttle = ego_action[0]  # continuous action: throttle
    steer = ego_action[1]  # continuous action: steering
    brake = ego_action[2]  # continuous action: brake

    # action range
    acc_max = acc_range[1]
    acc_min = acc_range[0]
    steering_max = steering_range[1]
    steering_min = steering_range[0]

    # normalize and clip the action

    throttle_max = acc_max / 3.
    throttle_min = acc_min / 3.
    throttle = np.clip(throttle, throttle_min, throttle_max)
    throttle = linear_map(throttle, [throttle_min, throttle_max], [0., 1.])

    steer = steer * steering_max
    steer = np.clip(steer, steering_min, steering_max)

    brake = np.clip(brake, -1., 1.)
    brake = linear_map(brake, [-1., 1.], [0., 1.])

    if brake < 0.05:
        brake = 0.0
    if throttle > brake:
        brake = 0.0

    return [throttle, steer, brake]


def get_feasibility_V(feasibility_policy, ego_obs):
    ego_obs = CUDA(torch.FloatTensor(ego_obs)).unsqueeze(0)
    V = feasibility_policy.get_feasibility_Vs(ego_obs).squeeze(0)
    return CPU(V).item()


def get_ego_obs(ego_vehicle, ego_nearby_vehicles, desired_nearby_vehicle=3, waypoints=None):
    '''
        safety network input state:
        all the rows are other bv's relative state
    '''
    infos = []
    # the basic information about the ego (center vehicle)
    ego_transform = CarlaDataProvider.get_transform(ego_vehicle)
    ego_matrix = np.array(ego_transform.get_matrix())
    ego_yaw = ego_transform.rotation.yaw / 180 * np.pi
    ego_extent = ego_vehicle.bounding_box.extent
    # the relative CBV info
    ego_info = get_relative_info(actor=ego_vehicle, center_yaw=ego_yaw, center_matrix=ego_matrix)
    infos.append(ego_info)
    for actor in ego_nearby_vehicles:
        if len(infos) < desired_nearby_vehicle:
            actor_info = get_relative_info(actor=actor, center_yaw=ego_yaw, center_matrix=ego_matrix)
            infos.append(actor_info)
        else:
            break
    while len(infos) < desired_nearby_vehicle:  # if no enough nearby vehicles, padding with 0
        infos.append([0] * len(ego_info))

    # route information
    if waypoints is not None:
        route_info = get_relative_route_info(waypoints, center_yaw=ego_yaw, center_matrix=ego_matrix, center_extent=ego_extent)
        infos.append(route_info)

    # get the info of the ego vehicle and the other actors
    infos = np.array(infos, dtype=np.float32)

    return {
        'ego_obs': infos
    }


def get_records(ego, CBVs_collision, ego_nearby_vehicles, goal_waypoint, goal_point_radius=2, search_radius=25, bbox=True, BV_num=3):
    # ego's info
    ego_vel = CarlaDataProvider.get_velocity(ego)
    ego_tra = CarlaDataProvider.get_transform(ego)
    ego_loc = ego_tra.location
    ego_acc = ego.get_acceleration()
    ego_extent = ego.bounding_box.extent
    # total record
    records = {'ego_vel': [ego_vel.x, ego_vel.y], 'ego_loc': [ego_loc.x, ego_loc.y], 'ego_id': ego.id, 'ego_extent': [ego_extent.x * 2., ego_extent.y * 2.],
               'ego_acc': [ego_acc.x, ego_acc.y], 'CBVs_collision': {}, 'BVs_loc': [], 'BVs_vel': [], 'BVs_yaw': [], 'BVs_extent': [], 'BVs_ego_dis': [], 'BVs_id': [],
               'CBVs_id': set(CarlaDataProvider.get_CBVs_by_ego(ego).keys()), 'goal_waypoint_loc': [goal_waypoint.transform.location.x, goal_waypoint.transform.location.y],
               'goal_radius': goal_point_radius, 'ego_min_dis': search_radius, 'ego_CBV_obs': {}}
    # get the ego CBV obs for feasibility calculation
    for CBV_id, CBV in CarlaDataProvider.get_CBVs_by_ego(ego).items():
        records['ego_CBV_obs'][CBV_id] = get_ego_obs(ego_vehicle=ego, ego_nearby_vehicles=[CBV])['ego_obs']
    # CBV's info
    if len(CBVs_collision) > 0:
        collision = {}
        for CBV_id, collision_event in CBVs_collision.items():
            # only record the collision between CBV and ego
            if collision_event is not None:
                collision[CBV_id] = {
                    'other_actor_id': collision_event['other_actor_id'],
                    'normal_impulse': collision_event['normal_impulse']
                }
            else:
                collision[CBV_id] = None
        records['CBVs_collision'] = collision

    # BVs info
    if ego_nearby_vehicles:
        BVs_ego_dis = []
        BVs_id = []
        BVs_abs_loc = []
        BVs_vel = []
        BVs_extent = []
        BVs_yaw = []

        for actor in ego_nearby_vehicles:
            if len(BVs_id) < BV_num:
                actor_transform = CarlaDataProvider.get_transform(actor)
                actor_vel = CarlaDataProvider.get_velocity(actor)
                BVs_vel.append([actor_vel.x, actor_vel.y])
                BVs_id.append(actor.id)
                loc = CarlaDataProvider.get_location(actor)
                BVs_abs_loc.append([loc.x, loc.y])
                actor_extent = actor.bounding_box.extent
                BVs_extent.append([actor_extent.x * 2., actor_extent.y * 2.])
                BVs_yaw.append(actor_transform.rotation.yaw / 180 * np.pi)
                BVs_ego_dis.append(get_min_distance_across_bboxes(ego, actor) if bbox else get_distance_across_centers(ego, actor))
            else:
                break
        BVs_abs_loc = np.array(BVs_abs_loc, dtype=np.float32)
        BVs_extent = np.array(BVs_extent, dtype=np.float32)
        BVs_vel = np.array(BVs_vel, dtype=np.float32)
        records['BVs_loc'] = BVs_abs_loc
        records['BVs_vel'] = BVs_vel
        records['BVs_yaw'] = BVs_yaw
        records['BVs_extent'] = BVs_extent
        records['BVs_ego_dis'] = BVs_ego_dis
        records['BVs_id'] = BVs_id
        records['ego_min_dis'] = min(BVs_ego_dis)

    return records


def draw_trajectory(world, ego, time, life_time=4):
    ego_color = carla.Color(200, 0, 0, 0)  # red
    CBV_color = carla.Color(100, 0, 250, 0)  # purple
    BV_color = carla.Color(0, 250, 100, 0)  # blue
    size = 0.3 if time % 15 == 0 else 0.1
    scenario_actor = CarlaDataProvider.get_scenario_actors_by_ego(ego)
    # draw ego trajectory
    ego_loc = CarlaDataProvider.get_location(ego)
    world.debug.draw_point(ego_loc + carla.Location(z=2), size=size, color=ego_color, life_time=life_time)
    # draw BVs trajectory
    for actor in scenario_actor['BVs'].values():
        actor_loc = CarlaDataProvider.get_location(actor)
        world.debug.draw_point(actor_loc + carla.Location(z=2), size=size, color=BV_color, life_time=life_time)

    # draw CBVs trajectory
    for actor in scenario_actor['CBVs'].values():
        actor_loc = CarlaDataProvider.get_location(actor)
        world.debug.draw_point(actor_loc + carla.Location(z=2), size=size, color=CBV_color, life_time=life_time)

    # draw CBVs reach goal trajectory
    for actor in scenario_actor['CBVs_reach_goal'].values():
        actor_loc = CarlaDataProvider.get_location(actor)
        world.debug.draw_point(actor_loc + carla.Location(z=2), size=size, color=CBV_color, life_time=life_time)


def get_ego_min_dis(ego, ego_nearby_vehicles, search_radius=25, bbox=True):
    ego_min_dis = search_radius
    if ego_nearby_vehicles:
        for i, vehicle in enumerate(ego_nearby_vehicles):
            if i < 3:  # calculate only the closest three vehicles
                dis = get_min_distance_across_bboxes(ego, vehicle) if bbox else get_distance_across_centers(ego, vehicle)
                if dis < ego_min_dis:
                    ego_min_dis = dis
    return ego_min_dis


def update_goal_CBV_dis(ego, goal_waypoint):
    """
        if the CBV has changed, then reset the corresponding initial distance
    """
    ego_id = ego.id
    # reset the ego CBV distance dict
    CarlaDataProvider.goal_CBV_dis[ego_id] = {}

    for CBV_id, CBV in CarlaDataProvider.get_CBVs_by_ego(ego).items():
        CBV_loc = CarlaDataProvider.get_location(CBV)
        CarlaDataProvider.goal_CBV_dis[ego_id][CBV_id] = CBV_loc.distance(goal_waypoint.transform.location)


def get_CBV_ego_reward(ego, CBV, goal_waypoint):
    """
        distance ratio and delta distance calculation
    """
    # get the dis between the goal and the CBV
    CBV_loc = CarlaDataProvider.get_location(CBV)
    dis = CBV_loc.distance(goal_waypoint.transform.location)

    # delta_dis > 0 means ego and CBV are getting closer, otherwise punish CBV drive away from ego
    delta_dis = np.clip(CarlaDataProvider.goal_CBV_dis[ego.id][CBV.id] - dis, a_min=-1., a_max=1.)

    return delta_dis, dis


def get_CBV_bv_reward(CBV, search_radius, CBV_nearby_vehicles, tou=1):
    min_dis = search_radius  # the searching radius of the nearby_vehicle
    if CBV and CBV_nearby_vehicles:
        for i, vehicle in enumerate(CBV_nearby_vehicles):
            if vehicle.attributes.get('role_name') == 'background' and i < 3:  # except the ego vehicle and calculate only the closest two vehicles
                # the min distance between bounding boxes of two vehicles
                min_dis = get_min_distance_across_bboxes(CBV, vehicle)
        min_dis_reward = min(min_dis, tou) - tou  # the controlled bv shouldn't be too close to the other bvs
    else:
        min_dis_reward = 0
    return min_dis, min_dis_reward


def get_locations_nearby_spawn_points(location_lists, radius_list, closest_dis, global_route_lane_road_ids, intensity=0.6, low_limit=5, upper_limit=18):
    CarlaDataProvider.generate_spawn_points()  # get all the possible spawn points in this map

    ego_locations = [ego.get_location() for ego in CarlaDataProvider.egos]
    ego_waypoint = CarlaDataProvider.get_map().get_waypoint(location=ego_locations[0], project_to_road=True)

    nearby_spawn_points = []
    for spawn_point in CarlaDataProvider._spawn_points:
        spawn_waypoint = CarlaDataProvider.get_map().get_waypoint(location=spawn_point.location, project_to_road=True)
        # remove the opposite lane vehicle on the same road
        if spawn_waypoint.road_id == ego_waypoint.road_id and spawn_waypoint.lane_id * ego_waypoint.lane_id < 0:
            continue

        # remove the waypoint in the global route
        if (spawn_waypoint.lane_id, spawn_waypoint.road_id) in global_route_lane_road_ids:
            continue

        in_radius = any(
            spawn_point.location.distance(location) <= radius
            for location, radius in zip(location_lists, radius_list)
        )

        far_from_ego = all(
            spawn_point.location.distance(ego_location) > closest_dis
            for ego_location in ego_locations
        )

        if in_radius and far_from_ego:
            nearby_spawn_points.append(spawn_point)

    # # debugging the location of all the spawn points
    # for point in nearby_spawn_points:
    #     CarlaDataProvider.get_world().debug.draw_point(point.location + carla.Location(z=2.0), size=0.1, color=carla.Color(0, 0, 255, 0), life_time=-1)

    CarlaDataProvider._rng.shuffle(nearby_spawn_points)
    spawn_points_count = len(nearby_spawn_points)
    if spawn_points_count < low_limit:
        picking_number = low_limit
    else:
        picking_number = max(min(int(spawn_points_count * intensity), upper_limit), low_limit)

    nearby_spawn_points = nearby_spawn_points[:picking_number]  # sampling part of the nearby spawn points

    return nearby_spawn_points


def find_closest_vehicle(ego_vehicle, radius=20, CBV_candidates=None):
    '''
        rule-based method to find the CBV:
        find the closest vehicle among all the CBV candidates
    '''
    min_dis = radius
    CBV = None
    ego_location = CarlaDataProvider.get_location(ego_vehicle)

    for vehicle in CBV_candidates:
        vehicle_location = CarlaDataProvider.get_location(vehicle)
        distance = ego_location.distance(vehicle_location)
        if distance < min_dis:
            CBV = vehicle  # update CBV
            min_dis = distance  # update min dis

    return CBV


def get_nearby_vehicles(center_vehicle, radius=20):
    '''
        return the nearby vehicles around the center vehicle
    '''
    center_location = CarlaDataProvider.get_location(center_vehicle)

    # get all the vehicles on the world using the actor dict on the CarlaDataProvider
    all_vehicles = CarlaDataProvider.get_actors()

    # store the nearby vehicle information [vehicle, distance]
    nearby_vehicles_info = []

    for vehicle_id, vehicle in all_vehicles.items():
        if vehicle_id != center_vehicle.id:  # except the center vehicle
            # the location of other vehicles
            vehicle_location = CarlaDataProvider.get_location(vehicle)
            distance = center_location.distance(vehicle_location)
            if distance <= radius:
                nearby_vehicles_info.append([vehicle, distance])

    # sort the nearby vehicles according to the distance in ascending order
    nearby_vehicles_info.sort(key=lambda x: x[1])

    # return the nearby vehicles list
    nearby_vehicles = [info[0] for info in nearby_vehicles_info]

    return nearby_vehicles


def normalize_angle(x):
    x = x % (2 * np.pi)  # force in range [0, 2 pi)
    if x > np.pi:  # move to [-pi, pi)
        x -= 2 * np.pi
    return x


def get_forward_speed(transform, velocity):
    """
        Convert the vehicle transform directly to forward speed
    """

    vel_np = np.array([velocity.x, velocity.y, velocity.z])
    pitch = np.deg2rad(transform.rotation.pitch)
    yaw = np.deg2rad(transform.rotation.yaw)
    orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    speed = np.dot(vel_np, orientation)
    return speed


def get_relative_transform(ego_matrix, vehicle_matrix):
    """
    return the relative transform from ego_pose to vehicle pose
    """
    relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
    rot = ego_matrix[:3, :3].T
    relative_pos = rot @ relative_pos

    # transform to the right-handed system
    relative_pos[1] = - relative_pos[1]

    return relative_pos


def get_relative_waypoint_info(goal_waypoint, center_yaw, center_matrix, radius=2):
    """
        get the relative waypoint info from the view of center vehicle
        info [x, y, 0, 0, yaw, 0]
    """
    goal_transform = goal_waypoint.transform
    goal_matrix = np.array(goal_transform.get_matrix())
    # relative yaw angle
    yaw = goal_transform.rotation.yaw / 180 * np.pi
    relative_yaw = normalize_angle(yaw - center_yaw)
    # relative pos
    relative_pos = get_relative_transform(ego_matrix=center_matrix, vehicle_matrix=goal_matrix)
    distance = np.linalg.norm(relative_pos)
    goal_info = [relative_pos[0], relative_pos[1], radius, radius, relative_yaw, distance]

    return goal_info


def get_relative_route_info(waypoints, center_yaw, center_matrix, center_extent):
    """
        get the relative route info from the view of center vehicle
        info [x, y, bbox_x, bbox_y, yaw, distance]
    """
    waypoint_route = np.array([[node[0], node[1]] for node in waypoints])
    max_len = 12
    if len(waypoint_route) < max_len:
        max_len = len(waypoint_route)
    shortened_route = rdp(waypoint_route[:max_len], epsilon=0.5)

    # convert points to vectors
    vectors = shortened_route[1:] - shortened_route[:-1]
    midpoints = shortened_route[:-1] + vectors / 2.
    norms = np.linalg.norm(vectors, axis=1)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])

    i = 0  # only use the first midpoint
    midpoint = midpoints[i]
    # find distance to center of waypoint
    center_bounding_box = carla.Location(midpoint[0], midpoint[1], 0.0)
    transform = carla.Transform(center_bounding_box)
    route_matrix = np.array(transform.get_matrix())
    relative_pos = get_relative_transform(center_matrix, route_matrix)
    distance = np.linalg.norm(relative_pos)

    length_bounding_box = carla.Vector3D(norms[i] / 2., center_extent.y, center_extent.z)
    bounding_box = carla.BoundingBox(transform.location, length_bounding_box)
    bounding_box.rotation = carla.Rotation(pitch=0.0,
                                           yaw=angles[i] * 180 / np.pi,
                                           roll=0.0)

    route_extent = bounding_box.extent
    dx = np.array([route_extent.x, route_extent.y, route_extent.z]) * 2.
    relative_yaw = normalize_angle(angles[i] - center_yaw)

    route_info = [relative_pos[0], relative_pos[1], dx[0], dx[1], relative_yaw, distance]

    return route_info


def get_relative_info(actor, center_yaw, center_matrix):
    """
        get the relative actor info from the view of center vehicle
        info [x, y, bbox_x, bbox_y, yaw, forward speed]
    """
    actor_transform = CarlaDataProvider.get_transform(actor)
    actor_rotation = actor_transform.rotation
    actor_matrix = np.array(actor_transform.get_matrix())
    # actor bbox
    actor_extent = actor.bounding_box.extent
    dx = np.array([actor_extent.x, actor_extent.y]) * 2.
    # relative yaw angle
    yaw = actor_rotation.yaw / 180 * np.pi
    relative_yaw = normalize_angle(yaw - center_yaw)
    # relative pos
    relative_pos = get_relative_transform(ego_matrix=center_matrix, vehicle_matrix=actor_matrix)
    actor_velocity = CarlaDataProvider.get_velocity(actor)
    actor_speed = get_forward_speed(transform=actor_transform, velocity=actor_velocity)  # In m/s
    actor_info = [relative_pos[0], relative_pos[1], dx[0], dx[1], relative_yaw, actor_speed]

    return actor_info


def check_CBV_BV_stuck(CBV, ego, max_distance=10, angle=10):
    CBV_BV_stuck = False

    CBV_trans = CarlaDataProvider.get_transform(CBV)
    for old_CBV in CarlaDataProvider.get_CBVs_reach_goal_by_ego(ego).values():
        old_CBV_trans = CarlaDataProvider.get_transform(old_CBV)
        if is_within_distance_ahead(CBV_trans, old_CBV_trans, max_distance, angle):
            CBV_BV_stuck = True
            break

    return CBV_BV_stuck


def check_interaction(ego, CBV, ego_length, delta_forward_angle=90, ego_fov=180):
    ego_transform = CarlaDataProvider.get_transform(ego)
    ego_forward_vector = ego_transform.rotation.get_forward_vector()
    CBV_transform = CarlaDataProvider.get_transform(CBV)
    CBV_forward_vector = CBV_transform.rotation.get_forward_vector()
    interaction = True
    ego_location = ego_transform.location
    CBV_location = CBV_transform.location
    relative_direction = (CBV_location - ego_location)
    relative_delta_angle = math.degrees(ego_forward_vector.get_vector_angle(relative_direction))
    distance = ego_location.distance(CBV_location)
    direction_angle = math.degrees(ego_forward_vector.get_vector_angle(CBV_forward_vector))

    # 1. ego and CBV got different direction, and behind ego a certain distance
    if distance >= ego_length and relative_delta_angle >= ego_fov / 2 and direction_angle > delta_forward_angle:
        interaction = False

    return interaction


def get_CBV_candidates(ego_vehicle, goal_waypoint, search_radius, ego_length):
    """
        the foundation for the CBV selection, selecting the candidates nearby vehicles based on specific traffic rules
        ego_vehicle: the ego vehicle
    """
    # info for the ego vehicle
    ego_transform = CarlaDataProvider.get_transform(ego_vehicle)
    ego_location = ego_transform.location
    ego_waypoint = CarlaDataProvider.get_map().get_waypoint(location=ego_location, project_to_road=True)
    goal_waypoint_lane_id = goal_waypoint.lane_id

    # get all the vehicles on the world to use the actor pool in CarlaDataProvider
    all_actors = CarlaDataProvider.get_actors()
    candidates = {key: value for key, value in all_actors.items()}
    # 1. remove the ego vehicle and the already reached goal CBV
    candidates.pop(ego_vehicle.id, None)
    for CBV_reach_goal_id in CarlaDataProvider.get_CBVs_reach_goal_by_ego(ego_vehicle).keys():
        candidates.pop(CBV_reach_goal_id, None)

    key_to_remove = []
    for vehicle_id, vehicle in candidates.items():
        vehicle_location = CarlaDataProvider.get_location(vehicle)
        vehicle_waypoint = CarlaDataProvider.get_map().get_waypoint(location=vehicle_location, project_to_road=True)

        # 2. remove the too far away vehicle
        if ego_location.distance(vehicle_location) > search_radius:
            key_to_remove.append(vehicle_id)
            continue

        # 3. if the ego waypoint and the goal waypoint are all not in the junction, remove the vehicle on the opposite lane
        if (not goal_waypoint.is_junction) or (not ego_waypoint.is_junction):
            if vehicle_waypoint.lane_id * goal_waypoint_lane_id < 0:
                key_to_remove.append(vehicle_id)
                continue

        # 4. remove the back vehicle with no interaction
        if not check_interaction(ego_vehicle, vehicle, ego_length, delta_forward_angle=90, ego_fov=180):
            # hard condition to check CBV candidates
            key_to_remove.append(vehicle_id)
            continue

        if check_CBV_BV_stuck(vehicle, ego_vehicle, max_distance=12, angle=15):
            key_to_remove.append(vehicle_id)
            continue

    for key in key_to_remove:
        del candidates[key]

    return list(candidates.values())


def get_min_distance_across_bboxes(veh1, veh2):
    box2origin_veh1, size_veh1 = compute_box2origin(veh1.bounding_box, CarlaDataProvider.get_transform(veh1))
    box2origin_veh2, size_veh2 = compute_box2origin(veh2.bounding_box, CarlaDataProvider.get_transform(veh2))
    # min distance
    box_collider_veh1 = colliders.Box(box2origin_veh1, size_veh1)
    box_collider_veh2 = colliders.Box(box2origin_veh2, size_veh2)
    dist, closest_point_box, closest_point_box2, _ = gjk.gjk(
        box_collider_veh1, box_collider_veh2)
    return dist


def get_distance_across_centers(veh1, veh2):
    veh1_loc = CarlaDataProvider.get_location(veh1)
    veh2_loc = CarlaDataProvider.get_location(veh2)
    return veh1_loc.distance(veh2_loc)


def calculate_abs_velocity(velocity):
    return round(math.sqrt(velocity.x ** 2 + velocity.y ** 2), 2)


def calculate_abs_acc(acc):
    return round(math.sqrt(acc.x ** 2 + acc.y ** 2), 2)
