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
from eval_analysis.process_data.feasibility import get_overall_unfeasible_ratio


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
                        'ego_CBV_obs': [],
                    }
                    # reverse the trajectory
                    for i in range(index, -1, -1):
                        if CBV_id in sequence[i]['CBVs_id_set'] and CBV_id in sequence[i]['BVs_id_set']:
                            BV_current_index = sequence[i]['BVs_id'].index(CBV_id)
                            if sequence[i]['BVs_ego_dis'][BV_current_index] < 25:
                                goal_reached_trajectories[CBV_id]['time'].append(sequence[i]['current_game_time'])
                                goal_reached_trajectories[CBV_id]['loc'].append(sequence[i]['BVs_loc'][BV_current_index])
                                goal_reached_trajectories[CBV_id]['ego_dis'].append(sequence[i]['BVs_ego_dis'][BV_current_index])
                                goal_reached_trajectories[CBV_id]['extent'].append(sequence[i]['BVs_extent'][BV_current_index])
                                goal_reached_trajectories[CBV_id]['yaw'].append(sequence[i]['BVs_yaw'][BV_current_index])
                                goal_reached_trajectories[CBV_id]['vel'].append(sequence[i]['BVs_vel'][BV_current_index])
                                goal_reached_trajectories[CBV_id]['ego_CBV_obs'].append(sequence[i]['ego_CBV_obs'][CBV_id])
                        else:
                            break
    return goal_reached_trajectories


def process_miss_trajectory_from_one_pkl(pkl_path, algorithm, save_folder, feasibility_policy):
    data = joblib.load(pkl_path)

    PET = []
    ego_dis = []
    TTC = []
    reach_goal_trajectories = []
    for sequence in tqdm(data.values()):
        # use all BV trajectories
        trajectory = get_all_BV_trajectory(sequence)
        # get PET per trajectory
        PET.extend(get_trajectory_pet(trajectory))
        # get ego dis per trajectory
        ego_dis.extend(get_ego_dis(trajectory))
        # get TTC per trajectory
        TTC.extend(get_trajectory_ttc(trajectory))

        # use CBV reach goal trajectory
        reach_goal_trajectory = get_CBV_goal_reached_trajectory(sequence)
        reach_goal_trajectories.append(reach_goal_trajectory)

    miss_traj_info = {
        'PET': PET,
        'ego_dis': ego_dis,
        'TTC': TTC,
    }

    if 'standard' not in algorithm:
        # get overall unfeasible_ratio
        unfeasible_ratio, feasibility_Vs = get_overall_unfeasible_ratio(reach_goal_trajectories, feasibility_policy)
        miss_traj_info['unfeasible_ratio'] = unfeasible_ratio
        miss_traj_info['feasibility_Vs'] = feasibility_Vs

    # save ego min dis
    with open(osp.join(save_folder, "miss_traj_info.pkl"), 'wb') as pickle_file:
        pickle.dump(miss_traj_info, pickle_file)


