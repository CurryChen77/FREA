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

from eval_analysis.process_data.feasibility import get_all_infeasible_ratio


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


def process_goal_reach_trajectory_from_one_pkl(pkl_path, algorithm, save_folder, feasibility_policy):
    if 'standard' not in algorithm:
        data = joblib.load(pkl_path)

        reach_goal_trajectories = []
        for sequence in tqdm(data.values()):
            # use CBV reach goal trajectory
            reach_goal_trajectory = get_CBV_goal_reached_trajectory(sequence)
            reach_goal_trajectories.append(reach_goal_trajectory)

        goal_reach_traj_info = {}

        # save data
        with open(osp.join(save_folder, "goal_reach_traj_info.pkl"), 'wb') as pickle_file:
            pickle.dump(goal_reach_traj_info, pickle_file)


