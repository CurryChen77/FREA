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
from collections import Counter
from tqdm import tqdm
import joblib
import numpy as np
from safebench.scenario.scenario_definition.atomic_criteria import Status


def trajectory_infos(trajectory, index, sequence, CBV_id):
    """
        record the necessary information for trajectory
    """
    # reverse from the current step for each CBV ego collision event
    trajectory[CBV_id] = {
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
            trajectory[CBV_id]['loc'].append(sequence[i]['BVs_loc'][BV_index])
            trajectory[CBV_id]['vel'].append(sequence[i]['BVs_vel'][BV_index])
            trajectory[CBV_id]['yaw'].append(sequence[i]['BVs_yaw'][BV_index])
            trajectory[CBV_id]['time'].append(sequence[i]['current_game_time'])
            trajectory[CBV_id]['ego_dis'].append(sequence[i]['BVs_ego_dis'][BV_index])
            if sequence[i]['BVs_id'][0] in sequence[i]['CBVs_id_set']:
                trajectory[CBV_id]['ego_obs'].append(sequence[i]['ego_obs'])
            else:
                trajectory[CBV_id]['ego_obs'].append(None)
        else:
            break


def get_collision_trajectory(sequence):
    """
        convert each sequence trajectory to potential collision trajectory
        sequence: a complete trajectory for one small route scenario
    """
    # convert id list to set
    for step in sequence:
        step['CBVs_id_set'] = set(step['CBVs_id'])
        step['BVs_id_set'] = set(step['BVs_id'])

    collision_trajectories = {}
    for index, step in enumerate(sequence):
        for CBV_id, collision_event in step['CBVs_collision'].items():
            if collision_event is not None and collision_event['other_actor_id'] == step['ego_id']:
                if CBV_id not in collision_trajectories:
                    collision_trajectories[CBV_id] = []
                    # only count the first time CBV_id appear in the CBVs_collision
                    trajectory_infos(collision_trajectories, index, sequence, CBV_id)
    return collision_trajectories


def get_goal_reached_trajectory(sequence):
    """
        get all the CBV trajectories that reach the goal
    """
    # convert id list to set
    for step in sequence:
        step['CBVs_id_set'] = set(step['CBVs_id'])
        step['BVs_id_set'] = set(step['BVs_id'])
        step['CBVs_reached_goal_id_set'] = set(step['CBVs_reached_goal_ids'])

    goal_reached_trajectories = {}
    for index, step in enumerate(sequence):
        for CBV_id in step['CBVs_reached_goal_id_set']:
            # only count the first time of CBV reach goal
            if CBV_id not in goal_reached_trajectories:
                goal_reached_trajectories[CBV_id] = []
                # only get info for the first time CBV_id appear in CBV_reach_goal_ids
                trajectory_infos(goal_reached_trajectories, index-1, sequence, CBV_id)
    return goal_reached_trajectories


def process_collision_from_one_pkl(pkl_path, algorithm, save_folder):
    """
        collision rate:
        the collision num / route scenario num (for PPO-based CBV, the collision rate may > 1)
    """
    data = joblib.load(pkl_path)
    collision = {}
    collision_impulse = []
    if 'standard' in algorithm:
        # rule-based scenario agent
        num_collision = 0
        for sequence in tqdm(data.values()):  # for each data id (small scenario)
            for step in sequence:  # count all the collision event along the trajectory
                if step['collision'][0] == Status.FAILURE:
                    num_collision += 1
                    if step['collision'][2] is not None:
                        step_collision_impulse = step['collision'][2]
                        collision_impulse.append(np.sqrt(np.sum(np.square(np.array(step_collision_impulse) / 1000))))  # kN*s
        collision['collision_rate'] = num_collision / len(data)
        collision['collision_impulse'] = collision_impulse
    else:
        # learnable scenario agent
        num_collision = 0
        num_scenario = 0
        for sequence in tqdm(data.values()):
            num_scenario += 1
            for step in sequence:
                for CBV_id, collision_event in step['CBVs_collision'].items():
                    if collision_event is not None and collision_event['other_actor_id'] == step['ego_id']:
                        # only count the collision with ego vehicle
                        step_collision_impulse = collision_event['normal_impulse']
                        collision_impulse.append(np.sqrt(np.sum(np.square(np.array(step_collision_impulse) / 1000))))  # kN*s
                        num_collision += 1
                        break

        collision['collision_rate'] = num_collision / num_scenario
        collision['collision_impulse'] = collision_impulse
    # save Vehicle forward speed
    with open(osp.join(save_folder, "collision.pkl"), 'wb') as pickle_file:
        pickle.dump(collision, pickle_file)


def process_common_data_from_one_pkl(pkl_path, save_folder):
    total_step = 0
    near_count = 0
    min_dis = []
    data = joblib.load(pkl_path)
    for sequence in tqdm(data.values()):
        get_collision_trajectory(sequence)
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


