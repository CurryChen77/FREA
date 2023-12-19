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

import numpy as np
from distance3d import gjk, colliders
from safebench.scenario.tools.scenario_utils import compute_box2origin
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider


def get_actor_in_drivable_area(actor):
    actor_transform = CarlaDataProvider.get_transform(actor)
    actor_location = actor_transform.location
    # move the actor waypoint to the front part of the actor
    actor_bbox_extent_x = actor.bounding_box.extent.x
    theta = math.radians(actor_transform.rotation.yaw)
    delta_x = actor_bbox_extent_x // 2 * math.cos(theta)
    delta_y = actor_bbox_extent_x // 2 * math.sin(theta)
    actor_front_location = actor_location + carla.Location(x=delta_x, y=delta_y, z=0.0)
    # find the nearest waypoint around the actor front location
    actor_waypoint = CarlaDataProvider.get_map().get_waypoint(actor_front_location)
    if actor_waypoint.lane_type == carla.LaneType.Driving:
        in_drivable_area = 0
    else:
        in_drivable_area = -1
    return in_drivable_area


def get_constrain_h(ego_vehicle, search_radius, nearby_vehicles, ego_agent_learnable=False):
    # min distance between vehicle bboxes
    ego_min_dis = search_radius

    # the closest vehicle using center points distance may change when using bboxes distance
    if nearby_vehicles:
        for i, vehicle in enumerate(nearby_vehicles):
            # the closest vehicle using center point may not be the closest vehicle using bboxs
            if i < 3:
                dis = get_min_distance_across_bboxes(ego_vehicle, vehicle)
                if dis < ego_min_dis:
                    ego_min_dis = dis

    if ego_agent_learnable:  # TODO rule-based agent usually will not drive out the road, will learning-based method could
        # check whether the ego has reached the un-drivable area
        check_lane_type_list = [carla.LaneType.Driving]
        ego_transform = CarlaDataProvider.get_transform(ego_vehicle)
        ego_location = ego_transform.location
        # move the ego waypoint to the front part of the vehicle
        ego_bbox_extent_x = ego_vehicle.bounding_box.extent.x
        theta = math.radians(ego_transform.rotation.yaw)
        delta_x = ego_bbox_extent_x // 2 * math.cos(theta)
        delta_y = ego_bbox_extent_x // 2 * math.sin(theta)
        ego_front_location = ego_location + carla.Location(x=delta_x, y=delta_y, z=0.0)
        # find the nearest waypoint around the ego front location
        ego_waypoint = CarlaDataProvider.get_map().get_waypoint(ego_front_location, lane_type=carla.LaneType.Any)
        # for viz
        # CarlaDataProvider._world.debug.draw_point(ego_waypoint.transform.location + carla.Location(z=4), size=0.1, life_time=0.11)
        # if the ego is not in the drivable area (off the road) and not in the junction, then the ego min distance is set to be 0
        if (ego_waypoint.lane_type not in check_lane_type_list) and (not ego_waypoint.is_junction):
            print("ego outside the drivable area, ego min distance set to 0")
            ego_min_dis = 0.

    constrain_h = ego_min_dis
    return constrain_h


def get_ego_min_dis(ego, ego_nearby_vehicles, search_redius=40):
    ego_min_dis = search_redius
    if ego_nearby_vehicles:
        for i, vehicle in enumerate(ego_nearby_vehicles):
            if i < 3:  # calculate only the closest three vehicles
                dis = get_min_distance_across_bboxes(ego, vehicle)
                if dis < ego_min_dis:
                    ego_min_dis = dis
    return ego_min_dis


def reset_ego_cbv_dis(ego, cbv):
    if cbv:
        CarlaDataProvider.ego_cbv_dis[ego.id] = {}
        dis = get_distance_across_centers(ego, cbv)
        CarlaDataProvider.ego_cbv_dis[ego.id][cbv.id] = dis
    else:
        CarlaDataProvider.ego_cbv_dis[ego.id] = {}


def get_cbv_stuck_reward(cbv, cbv_nearby_vehicles, ego, ego_nearby_vehicles):
    """
        if cbv movement causing stuck in the traffic flow especially for ego, punish this
    """
    stuck_reward = 0
    if cbv is not None and cbv_nearby_vehicles is not None and ego_nearby_vehicles is not None:
        cbv_v = CarlaDataProvider.get_velocity(cbv)
        ego_v = CarlaDataProvider.get_velocity(ego)
        relative_velocity_list = [cbv_v.distance_2d(ego_v)]
        for bv in ego_nearby_vehicles:
            if any(actor.id == bv.id for actor in cbv_nearby_vehicles):
                bv_v = CarlaDataProvider.get_velocity(bv)
                relative_velocity_list.append(bv_v.distance_2d(ego_v))
                break
        if np.average(relative_velocity_list) < 0.1:
            stuck_reward = -1

    return stuck_reward


def get_ego_cbv_dis_reward(ego, cbv):
    '''
        if the cbv are getting closer to the ego vehicle, then we should reward that, else punish that
    '''
    delta_dis = 0
    if cbv:
        if cbv.id in CarlaDataProvider.ego_cbv_dis[ego.id].keys():  # whether the current are in the old cbv list
            dis = get_distance_across_centers(ego, cbv)
            # delta_dis > 0 means ego and cbv are getting closer, otherwise punish cbv drive away from ego
            delta_dis = CarlaDataProvider.ego_cbv_dis[ego.id][cbv.id] - dis
            CarlaDataProvider.ego_cbv_dis[ego.id][cbv.id] = dis
        else:
            CarlaDataProvider.ego_cbv_dis[ego.id] = {}
            dis = get_distance_across_centers(ego, cbv)
            CarlaDataProvider.ego_cbv_dis[ego.id][cbv.id] = dis
    else:
        CarlaDataProvider.ego_cbv_dis[ego.id] = {}
    return delta_dis


def get_cbv_bv_reward(cbv, search_radius, cbv_nearby_vehicles, tou=1.25):
    min_dis = search_radius  # the searching radius of the nearby_vehicle
    if cbv and cbv_nearby_vehicles:
        for i, vehicle in enumerate(cbv_nearby_vehicles):
            if vehicle.attributes.get('role_name') == 'background' and i < 3:  # except the ego vehicle and calculate only the closest two vehicles
                # the min distance between bounding boxes of two vehicles
                min_dis = get_min_distance_across_bboxes(cbv, vehicle)
        min_dis_reward = min(min_dis, tou) - tou  # the controlled bv shouldn't be too close to the other bvs
    else:
        min_dis_reward = 0
    return min_dis, min_dis_reward


def get_ego_cbv_reward(ego, cbv, affected_range=10):
    """
        dis reward ~ [0, affected_range]: within h
    """
    if cbv:
        ego_cbv_dis = get_distance_across_centers(ego, cbv)
        dis_reward = affected_range - min(ego_cbv_dis, affected_range)
    else:
        dis_reward = 0
    return dis_reward


def get_locations_nearby_spawn_points(location_lists, radius_list=None, closest_dis=5, intensity=0.8, upper_limit=18):
    nearby_spawn_points = []
    CarlaDataProvider.generate_spawn_points()  # get all the possible spawn points in this map
    # TODO the initial position of the bv still got problems, may to close to the egos
    ego_locations = [ego.get_location() for ego in CarlaDataProvider.egos]
    for spawn_point in CarlaDataProvider._spawn_points:
        spawn_point_location = spawn_point.location
        # check whether any spawn point close to any egos' location
        close_to_ego = any(ego_loc.distance(spawn_point_location) <= closest_dis for ego_loc in ego_locations)

        if any(location.distance(spawn_point_location) <= radius for location, radius in zip(location_lists, radius_list)) and not close_to_ego:
            # if the spawn point is in any location's radius and not close to any egos' location, add them to the list
            nearby_spawn_points.append(spawn_point)

    CarlaDataProvider._rng.shuffle(nearby_spawn_points)
    spawn_points_count = len(nearby_spawn_points)
    picking_number = min(int(spawn_points_count * intensity), upper_limit) if spawn_points_count > upper_limit else spawn_points_count
    nearby_spawn_points = nearby_spawn_points[:picking_number]  # sampling part of the nearby spawn points
    return nearby_spawn_points


def find_closest_vehicle(ego_vehicle, radius=40, cbv_candidates=None):
    '''
        rule-based method to find the cbv:
        find the closest vehicle among all the cbv candidates
    '''
    min_dis = radius
    cbv = None
    ego_location = CarlaDataProvider.get_location(ego_vehicle)
    if cbv_candidates is None:
        cbv_candidates = get_cbv_candidates(ego_vehicle, radius)  # find the cbv candidates

    for vehicle in cbv_candidates:
        vehicle_location = CarlaDataProvider.get_location(vehicle)
        distance = ego_location.distance(vehicle_location)
        if distance < min_dis:
            cbv = vehicle  # update cbv
            min_dis = distance  # update min dis

    return cbv


def get_nearby_vehicles(center_vehicle, radius=40):
    '''
        return the nearby vehicles around the center vehicle
    '''
    center_location = CarlaDataProvider.get_location(center_vehicle)

    # get all the vehicles on the world
    all_vehicles = CarlaDataProvider._world.get_actors().filter('vehicle.*')

    # store the nearby vehicle information [vehicle, distance]
    nearby_vehicles_info = []

    for vehicle in all_vehicles:
        if vehicle.id != center_vehicle.id:  # except the center vehicle
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


def get_cbv_candidates(center_vehicle, radius=40):
    '''
        the foundation for the cbv selection, selecting the candidates nearby vehicles based on specific traffic rules
    '''
    # info for the ego vehicle
    center_transform = CarlaDataProvider.get_transform(center_vehicle)
    center_location = center_transform.location
    center_forward_vector = center_transform.rotation.get_forward_vector()

    # get all the vehicles on the world
    all_vehicles = CarlaDataProvider._world.get_actors().filter('vehicle.*')
    # store the nearby vehicle information [vehicle, distance]
    nearby_vehicles_info = []
    for vehicle in all_vehicles:
        if vehicle.id != center_vehicle.id:  # 1. except the center vehicle
            vehicle_transform = CarlaDataProvider.get_transform(vehicle)
            vehicle_location = vehicle_transform.location

            relative_direction = (vehicle_location - center_location)
            # if dot product > 0: vehicle in front of ego; if dot product < 0: vehicle at the back of ego
            dot_product = center_forward_vector.dot(relative_direction)
            # 2. remove the bv behind the center vehicle and got different direction
            if dot_product > 0.0 or abs(center_transform.rotation.yaw - vehicle_transform.rotation.yaw) < 45:
                distance = center_location.distance(vehicle_location)
                if distance <= radius:
                    nearby_vehicles_info.append([vehicle, distance])

    # sort the nearby vehicles according to the distance in ascending order
    nearby_vehicles_info.sort(key=lambda x: x[1])

    # return the nearby vehicles list
    nearby_vehicles = [info[0] for info in nearby_vehicles_info]

    return nearby_vehicles


def get_min_distance_across_bboxes(veh1, veh2):
    veh1_bbox = veh1.bounding_box
    veh2_bbox = veh2.bounding_box
    veh1_transform = CarlaDataProvider.get_transform(veh1)
    veh2_transform = CarlaDataProvider.get_transform(veh2)

    box2origin_veh1, size_veh1 = compute_box2origin(veh1_bbox, veh1_transform)
    box2origin_veh2, size_veh2 = compute_box2origin(veh2_bbox, veh2_transform)
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
    return round(math.sqrt(velocity.x**2 + velocity.y**2), 2)
