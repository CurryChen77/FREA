#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：process_common_state.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/4/1
"""
import os.path as osp
import pickle
import math
from tqdm import tqdm
import joblib
import numpy as np

from eval_analysis.process_data.PET import get_trajectory_pet
from eval_analysis.process_data.TTC import get_trajectory_ttc


def get_ego_dis(trajectory):
    ego_dis = []
    for vehicle, data in trajectory.items():
        if vehicle != 'ego':
            ego_dis.extend(data['ego_dis'])
    return ego_dis


def get_standard_closest_BV_trajectory(sequence):
    """
        get all the closest BV trajectories
    """
    # convert id list to set
    for step in sequence:
        step['BVs_id_set'] = set(step['BVs_id'])

    closest_BV_trajectories = {'ego': {
        'time': [],
        'loc': [],
        'extent': [],
        'yaw': [],
        'vel': [],
    }}
    BV_closest_index = {}
    for index, step in enumerate(sequence):
        # store the ego info every step
        closest_BV_trajectories['ego']['time'].append(step['current_game_time'])
        closest_BV_trajectories['ego']['loc'].append(step['ego_loc'])
        closest_BV_trajectories['ego']['extent'].append(step['ego_extent'])
        closest_BV_trajectories['ego']['yaw'].append(step['ego_yaw'])
        closest_BV_trajectories['ego']['vel'].append(step['ego_vel'])
        for BV_id in step['BVs_id']:
            BV_index = step['BVs_id'].index(BV_id)
            BV_dis = step['BVs_ego_dis'][BV_index]
            if BV_dis < 5:
                # first add the closest index
                if BV_id not in BV_closest_index:
                    BV_closest_index[BV_id] = [index, BV_dis]
                # update the closest index
                elif BV_id in BV_closest_index and BV_dis < BV_closest_index[BV_id][1]:
                    BV_closest_index[BV_id] = [index, BV_dis]

    for BV_id, index_list in BV_closest_index.items():
        index = index_list[0]
        # create a new key for CBV reach goal
        closest_BV_trajectories[BV_id] = {
            'time': [],
            'loc': [],
            'ego_dis': [],
            'extent': [],
            'yaw': [],
            'vel': [],
        }
        # reverse the trajectory
        for i in range(index, -1, -1):
            if BV_id in sequence[i]['BVs_id_set']:
                BV_current_index = sequence[i]['BVs_id'].index(BV_id)
                if sequence[i]['BVs_ego_dis'][BV_current_index] < 15:
                    closest_BV_trajectories[BV_id]['time'].append(sequence[i]['current_game_time'])
                    closest_BV_trajectories[BV_id]['loc'].append(sequence[i]['BVs_loc'][BV_current_index])
                    closest_BV_trajectories[BV_id]['ego_dis'].append(sequence[i]['BVs_ego_dis'][BV_current_index])
                    closest_BV_trajectories[BV_id]['extent'].append(sequence[i]['BVs_extent'][BV_current_index])
                    closest_BV_trajectories[BV_id]['yaw'].append(sequence[i]['BVs_yaw'][BV_current_index])
                    closest_BV_trajectories[BV_id]['vel'].append(sequence[i]['BVs_vel'][BV_current_index])
            else:
                break
    return closest_BV_trajectories


def get_CBV_reach_goal(goal_loc, CBV_loc, goal_radius):
    """
        whether the CBV has reached the goal
    """
    return math.sqrt((goal_loc[0] - CBV_loc[0]) ** 2 + (goal_loc[1] - CBV_loc[1]) ** 2) <= goal_radius


def get_CBV_goal_reached_trajectory(sequence):
    """
        get all the CBV trajectories that reach the goal
    """
    # convert id list to set
    for step in sequence:
        step['CBVs_id_set'] = set(step['CBVs_id'])
        step['BVs_id_set'] = set(step['BVs_id'])

    goal_reached_trajectories = {'ego': {
        'time': [],
        'loc': [],
        'extent': [],
        'yaw': [],
        'vel': [],
    }}
    for index, step in enumerate(sequence):
        # store the ego info every step
        goal_reached_trajectories['ego']['time'].append(step['current_game_time'])
        goal_reached_trajectories['ego']['loc'].append(step['ego_loc'])
        goal_reached_trajectories['ego']['extent'].append(step['ego_extent'])
        goal_reached_trajectories['ego']['yaw'].append(step['ego_yaw'])
        goal_reached_trajectories['ego']['vel'].append(step['ego_vel'])
        for CBV_id in step['CBVs_id']:
            if CBV_id in step['BVs_id_set']:
                BV_index = step['BVs_id'].index(CBV_id)
                # if the CBV_id is the first time reach the goal
                if get_CBV_reach_goal(step['goal_waypoint_loc'], step['BVs_loc'][BV_index], step['goal_radius']) and CBV_id not in goal_reached_trajectories:
                    # create a new key for CBV reach goal
                    goal_reached_trajectories[CBV_id] = {
                        'time': [],
                        'loc': [],
                        'ego_dis': [],
                        'extent': [],
                        'yaw': [],
                        'vel': [],
                    }
                    # reverse the trajectory
                    for i in range(index, -1, -1):
                        if CBV_id in sequence[i]['CBVs_id_set'] and CBV_id in sequence[i]['BVs_id_set']:
                            BV_current_index = sequence[i]['BVs_id'].index(CBV_id)
                            if sequence[i]['BVs_ego_dis'][BV_current_index] < 15:
                                goal_reached_trajectories[CBV_id]['time'].append(sequence[i]['current_game_time'])
                                goal_reached_trajectories[CBV_id]['loc'].append(sequence[i]['BVs_loc'][BV_current_index])
                                goal_reached_trajectories[CBV_id]['ego_dis'].append(sequence[i]['BVs_ego_dis'][BV_current_index])
                                goal_reached_trajectories[CBV_id]['extent'].append(sequence[i]['BVs_extent'][BV_current_index])
                                goal_reached_trajectories[CBV_id]['yaw'].append(sequence[i]['BVs_yaw'][BV_current_index])
                                goal_reached_trajectories[CBV_id]['vel'].append(sequence[i]['BVs_vel'][BV_current_index])
                        else:
                            break
    return goal_reached_trajectories


def process_miss_trajectory_from_one_pkl(pkl_path, algorithm, save_folder):
    data = joblib.load(pkl_path)
    if 'standard' in algorithm:
        trajectory_function = get_standard_closest_BV_trajectory
    else:
        trajectory_function = get_CBV_goal_reached_trajectory

    PET = []
    ego_dis = []
    TTC = []
    for sequence in tqdm(data.values()):
        trajectory = trajectory_function(sequence)
        PET.extend(get_trajectory_pet(trajectory))
        ego_dis.extend(get_ego_dis(trajectory))
        TTC.extend(get_trajectory_ttc(trajectory))

    miss_traj_info = {
        'PET': PET,
        'ego_dis': ego_dis,
        'TTC': TTC,
    }

    # save ego min dis
    with open(osp.join(save_folder, "miss_traj_info.pkl"), 'wb') as pickle_file:
        pickle.dump(miss_traj_info, pickle_file)


