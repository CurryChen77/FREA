#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：overall_trajectory.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/7/28
"""
import os.path as osp
import numpy as np
import pickle
import torch
from tqdm import tqdm
import joblib
from frea.util.torch_util import CUDA, CPU
from eval_analysis.process_data.PET import get_trajectory_pet
from eval_analysis.process_data.TTC import get_trajectory_ttc
from eval_analysis.process_data.feasibility import get_all_infeasible_ratio, get_all_infeasible_distance
from distance3d import gjk, colliders


def compute_R(yaw):
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return R


def compute_box2origin_2D(vehicle_location, vehicle_yaw):
    t = np.array([
        vehicle_location[0],
        vehicle_location[1],
        0.755
    ])

    r = compute_R(vehicle_yaw)

    size = np.array([4.4, 1.8, 1.51])

    box2origin = np.zeros((4, 4))
    box2origin[:3, :3] = r
    box2origin[:3, 3] = t
    box2origin[3, 3] = 1.0

    return box2origin, size


def get_min_distance_across_centers(coords_1, coords_2):
    """
    Get the distance between two point coords -> [x, y]
    """
    return np.linalg.norm(np.array(coords_1) - np.array(coords_2))


def get_min_distance_across_boxes(coords_1, coords_2, yaw1, yaw2):
    box2origin_1, size_1 = compute_box2origin_2D(coords_1, yaw1)
    box2origin_2, size_2 = compute_box2origin_2D(coords_2, yaw2)
    box_collider_1 = colliders.Box(box2origin_1, size_1)
    box_collider_2 = colliders.Box(box2origin_2, size_2)
    dist, closest_point_box, closest_point_box2, _ = gjk.gjk(
        box_collider_1, box_collider_2)
    return dist


def create_yaw_rotation_matrix(yaw):
    """
    Create a 3x3 rotation matrix for a given yaw angle (in radians).
    """
    cos_theta = np.cos(yaw)
    sin_theta = np.sin(yaw)

    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])

    return rotation_matrix


def get_relative_transform(ego_coords, vehicle_coords, ego_yaw):
    """
    return the relative transform from ego_pose to vehicle pose
    """
    relative_pos = np.array(vehicle_coords + [0]) - np.array(ego_coords + [0])
    rot = create_yaw_rotation_matrix(ego_yaw).T
    relative_pos = rot @ relative_pos

    # transform to the right-handed system
    relative_pos[1] = - relative_pos[1]

    return relative_pos


def get_forward_speed_2d(yaw, velocity):
    """
    Convert the vehicle yaw directly to forward speed in 2D
    the velocity from King is []
    """
    vel_np = np.array([velocity[0], velocity[1]])
    orientation = np.array([np.cos(yaw), np.sin(yaw)])
    speed = np.dot(vel_np, orientation)
    return speed


def normalize_angle(x):
    x = x % (2 * np.pi)  # force in range [0, 2 pi)
    if x > np.pi:  # move to [-pi, pi)
        x -= 2 * np.pi
    return x


def get_relative_info(ego_coords, vehicle_coords, ego_yaw, vehicle_yaw, vehicle_velocity, extent):
    """
        get the relative actor info from the view of center vehicle
        info [x, y, bbox_x, bbox_y, yaw, forward speed]
    """
    # relative yaw angle
    relative_yaw = normalize_angle(vehicle_yaw - ego_yaw)
    # relative pos
    relative_pos = get_relative_transform(ego_coords=ego_coords, vehicle_coords=vehicle_coords, ego_yaw=ego_yaw)

    vehicle_forward_vel = get_forward_speed_2d(yaw=vehicle_yaw, velocity=vehicle_velocity)  # In m/s
    # the extent x and y from King setting
    vehicle_info = [relative_pos[0], relative_pos[1], extent[0], extent[1], relative_yaw, vehicle_forward_vel]

    return vehicle_info


def form_ego_obs(ego_info, vehicle_info, desired_nearby_vehicle=3):
    ego_obs = []
    ego_coords, ego_yaw, ego_velocity, ego_extent = ego_info
    ego_rel_info = get_relative_info(ego_coords=ego_coords, vehicle_coords=ego_coords, ego_yaw=ego_yaw, vehicle_yaw=ego_yaw, vehicle_velocity=ego_velocity, extent=ego_extent)
    ego_obs.append(ego_rel_info)

    vehicle_coords, vehicle_yaw, vehicle_velocity, vehicle_extent = vehicle_info
    vehicle_rel_info = get_relative_info(ego_coords=ego_coords, vehicle_coords=vehicle_coords, ego_yaw=ego_yaw, vehicle_yaw=vehicle_yaw, vehicle_velocity=vehicle_velocity, extent=ego_extent)
    ego_obs.append(vehicle_rel_info)
    ego_bv_dis = get_min_distance_across_boxes(ego_coords, vehicle_coords, ego_yaw, vehicle_yaw)
    # ego_bv_dis = get_min_distance_across_centers(ego_coords, vehicle_coords)

    while len(ego_obs) < desired_nearby_vehicle:
        ego_obs.append([0] * len(ego_rel_info))

    return np.array(ego_obs, dtype=np.float32), ego_bv_dis


def get_ego_dis(trajectory):
    ego_dis = []
    for vehicle, data in trajectory.items():
        if vehicle != 'ego':
            ego_dis.extend(data['ego_dis'])
    return ego_dis


def get_all_BV_trajectory(sequence):
    """
        get all the closest BV trajectories
    """
    # convert id list to set
    for step in sequence:
        step['BVs_id_set'] = set(step['BVs_id'])

    all_trajectories = {'ego': {
        'time': [],
        'loc': [],
        'extent': [],
        'yaw': [],
        'vel': [],
    }}
    for step in sequence:
        # store the ego info every step
        all_trajectories['ego']['time'].append(step['current_game_time'])
        all_trajectories['ego']['loc'].append(step['ego_loc'])
        all_trajectories['ego']['extent'].append(step['ego_extent'])
        all_trajectories['ego']['yaw'].append(step['ego_yaw'])
        all_trajectories['ego']['vel'].append(step['ego_vel'])
        for BV_id in step['BVs_id']:
            BV_index = step['BVs_id'].index(BV_id)
            if BV_id not in all_trajectories:
                # initialize trajectory info for a new BV
                all_trajectories[BV_id] = {
                    'time': [],
                    'loc': [],
                    'ego_dis': [],
                    'extent': [],
                    'yaw': [],
                    'vel': [],
                }
            else:
                # only record the BV within search radius
                if step['BVs_ego_dis'][BV_index] < 25:
                    all_trajectories[BV_id]['time'].append(step['current_game_time'])
                    all_trajectories[BV_id]['loc'].append(step['BVs_loc'][BV_index])
                    all_trajectories[BV_id]['ego_dis'].append(step['BVs_ego_dis'][BV_index])
                    all_trajectories[BV_id]['extent'].append(step['BVs_extent'][BV_index])
                    all_trajectories[BV_id]['yaw'].append(step['BVs_yaw'][BV_index])
                    all_trajectories[BV_id]['vel'].append(step['BVs_vel'][BV_index])

    return all_trajectories


def get_closest_trajectory(sequence, threshold=3):
    # convert id list to set
    for step in sequence:
        step['CBVs_id_set'] = set(step['CBVs_id'])
        step['BVs_id_set'] = set(step['BVs_id'])

    closest_index = {}
    # find the closest index for each CBV
    for index, step in enumerate(sequence):
        # update the closest index
        for CBV_id in step['CBVs_id']:
            if CBV_id in step['BVs_id_set']:
                BV_index = step['BVs_id'].index(CBV_id)
                BV_ego_dis = step['BVs_ego_dis'][BV_index]
                # Update the closest index and distance if necessary
                if CBV_id not in closest_index or BV_ego_dis < closest_index[CBV_id][0]:
                    closest_index[CBV_id] = [BV_ego_dis, index]

    closest_trajectories = {}
    for CBV_id, data in closest_index.items():
        # if min distance < threshold, then initialize the trajectory
        closest_dis, index = data
        if closest_dis < threshold:
            closest_trajectories[CBV_id] = {'ego_obs': [], 'ego_cbv_dis': []}
            # reverse the trajectory
            for i in range(index, -1, -1):
                if CBV_id in sequence[i]['BVs_id_set']:
                    BV_current_index = sequence[i]['BVs_id'].index(CBV_id)
                    if sequence[i]['BVs_ego_dis'][BV_current_index] < 25:
                        ego_info = [sequence[i]['ego_loc'], sequence[i]['ego_yaw'], sequence[i]['ego_vel'], sequence[i]['ego_extent']]
                        vehicle_loc = sequence[i]['BVs_loc'][BV_current_index]
                        vehicle_yaw = sequence[i]['BVs_yaw'][BV_current_index]
                        vehicle_vel = sequence[i]['BVs_vel'][BV_current_index]
                        vehicle_extent = sequence[i]['BVs_extent'][BV_current_index]
                        vehicle_info = [[vehicle_loc[0], vehicle_loc[1]], vehicle_yaw, [vehicle_vel[0], vehicle_vel[1]], [vehicle_extent[0], vehicle_extent[1]]]
                        ego_obs, ego_cbv_dis = form_ego_obs(ego_info, vehicle_info)
                        closest_trajectories[CBV_id]['ego_obs'].append(ego_obs)
                        closest_trajectories[CBV_id]['ego_cbv_dis'].append(ego_cbv_dis)
                else:
                    break
    return closest_trajectories


def process_all_trajectory_from_one_pkl(pkl_path, algorithm, save_folder, feasibility_policy):
    data = joblib.load(pkl_path)

    PET = []
    ego_dis = []
    TTC = []

    for sequence in tqdm(data.values()):
        # use all BV trajectories
        all_BV_trajectory = get_all_BV_trajectory(sequence)
        # get PET per trajectory
        PET.extend(get_trajectory_pet(all_BV_trajectory))
        # get ego dis per trajectory
        ego_dis.extend(get_ego_dis(all_BV_trajectory))
        # get TTC per trajectory
        TTC.extend(get_trajectory_ttc(all_BV_trajectory))

    all_traj_info = {
        'PET': PET,
        'ego_dis': ego_dis,
        'TTC': TTC,
    }

    # the feasible metric
    if 'standard' not in algorithm:
        closest_trajectory_list = []
        for sequence in tqdm(data.values()):
            # get the closest trajectory
            closest_trajectory_list.append(get_closest_trajectory(sequence))

        # get overall infeasible_ratio
        infeasible_ratio, feasibility_Vs = get_all_infeasible_ratio(closest_trajectory_list, feasibility_policy)
        infeasible_distance, all_infeasible_distance = get_all_infeasible_distance(closest_trajectory_list, feasibility_policy)
        # print("infeasible distance", infeasible_distance)
        # print("infeasible_ratio", infeasible_ratio)

        all_traj_info.update({
            'infeasible_ratio': infeasible_ratio,
            'feasibility_Vs': feasibility_Vs,
            'infeasible_distance': infeasible_distance,
            'all_infeasible_distance': all_infeasible_distance,
        })

    # save data
    with open(osp.join(save_folder, "all_traj_info.pkl"), 'wb') as pickle_file:
        pickle.dump(all_traj_info, pickle_file)
