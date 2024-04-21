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
from collections import Counter
from tqdm import tqdm
import joblib
import numpy as np


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

    goal_reached_trajectories = {}
    for index, step in enumerate(sequence):
        for CBV_id in step['CBVs_id']:
            if CBV_id in step['BVs_id_set']:
                BV_index = step['BVs_id'].index(CBV_id)
                if get_CBV_reach_goal(step['goal_waypoint_loc'], step['BVs_loc'][BV_index], step['goal_radius']):
                    # only count the first time of CBV reach goal
                    if CBV_id not in goal_reached_trajectories:
                        # only get info for the first time CBV_id appear in CBV_reach_goal_ids
                        goal_reached_trajectories[CBV_id] = {
                            'loc': [],
                            'vel': [],
                            'yaw': [],
                            'time': [],
                            'ego_dis': [],
                            'ego_obs': [],
                        }
                        for i in range(index, -1, -1):
                            if CBV_id in sequence[i]['CBVs_id_set'] and CBV_id in sequence[i]['BVs_id_set']:
                                BV_index = sequence[i]['BVs_id'].index(CBV_id)
                                goal_reached_trajectories[CBV_id]['loc'].append(sequence[i]['BVs_loc'][BV_index])
                                goal_reached_trajectories[CBV_id]['vel'].append(sequence[i]['BVs_vel'][BV_index])
                                goal_reached_trajectories[CBV_id]['yaw'].append(sequence[i]['BVs_yaw'][BV_index])
                                goal_reached_trajectories[CBV_id]['time'].append(sequence[i]['current_game_time'])
                                goal_reached_trajectories[CBV_id]['ego_dis'].append(sequence[i]['BVs_ego_dis'][BV_index])
                            else:
                                break
    return goal_reached_trajectories


def process_miss_trajectory_from_one_pkl(pkl_path, save_folder):
    total_step = 0
    near_count = 0
    min_dis = []
    data = joblib.load(pkl_path)
    for sequence in tqdm(data.values()):
        for step in sequence:
            if step['ego_min_dis'] < 25:
                total_step += 1

                min_dis.append(step['ego_min_dis'])
                near_count += 1 if step['ego_min_dis'] < 1 else 0

    min_dis_data = {
        'near_rate': near_count / total_step,
        'min_dis': min_dis
    }
    # save ego min dis
    with open(osp.join(save_folder, "min_dis.pkl"), 'wb') as pickle_file:
        pickle.dump(min_dis_data, pickle_file)


