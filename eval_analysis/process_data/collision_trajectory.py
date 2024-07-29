#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：collision_trajectory.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/4/18
"""
import os.path as osp
import numpy as np
import pickle
from tqdm import tqdm
import joblib

from eval_analysis.process_data.all_trajectory import form_ego_obs
from eval_analysis.process_data.feasibility import get_all_infeasible_ratio, get_all_infeasible_distance


def get_collision_trajectory(sequence):
    collision_trajectories = {}

    for index, step in enumerate(sequence):
        for CBV_id, collision_event in step['CBVs_collision'].items():
            # only count the CBV collision with the ego vehicle
            if collision_event is not None and collision_event['other_actor_id'] == step['ego_id']:
                collision_trajectories[CBV_id] = {'ego_obs': [], 'ego_cbv_dis': []}
                for i in range(index, -1, -1):
                    # reverse the whole trajectory of the collision CBV
                    if CBV_id in sequence[i]['CBVs_id'] and CBV_id in sequence[i]['BVs_id']:
                        BV_current_index = sequence[i]['BVs_id'].index(CBV_id)
                        # record the ego obs
                        ego_info = [sequence[i]['ego_loc'], sequence[i]['ego_yaw'], sequence[i]['ego_vel'], sequence[i]['ego_extent']]
                        vehicle_loc = sequence[i]['BVs_loc'][BV_current_index]
                        vehicle_yaw = sequence[i]['BVs_yaw'][BV_current_index]
                        vehicle_vel = sequence[i]['BVs_vel'][BV_current_index]
                        vehicle_extent = sequence[i]['BVs_extent'][BV_current_index]
                        vehicle_info = [[vehicle_loc[0], vehicle_loc[1]], vehicle_yaw, [vehicle_vel[0], vehicle_vel[1]], [vehicle_extent[0], vehicle_extent[1]]]
                        ego_obs, ego_cbv_dis = form_ego_obs(ego_info, vehicle_info)
                        collision_trajectories[CBV_id]['ego_obs'].append(ego_obs)
                        collision_trajectories[CBV_id]['ego_cbv_dis'].append(ego_cbv_dis)
                    else:
                        break

    return collision_trajectories


def get_CBV_collision_severity(sequence):
    CBV_vel = []
    collision_count = 0
    collision_impulse = []

    for index, step in enumerate(sequence):
        for CBV_id, collision_event in step['CBVs_collision'].items():
            # only count the CBV collision with the ego vehicle
            if collision_event is not None and collision_event['other_actor_id'] == step['ego_id']:
                # collision impulse
                collision_impulse.append(np.sqrt(np.sum(np.square(np.array(collision_event['normal_impulse']) / 1000))))  # kN*s
                collision_count += 1

                for i in range(index, max(0, index-4), -1):
                    # reverse the whole trajectory of the collision CBV
                    if CBV_id in sequence[i]['CBVs_id'] and CBV_id in sequence[i]['BVs_id']:
                        BV_index = sequence[i]['BVs_id'].index(CBV_id)
                        # record the velocity
                        CBV_vel.append(np.linalg.norm(np.array(sequence[i]['BVs_vel'][BV_index])))
                    else:
                        break

    return CBV_vel, collision_impulse, collision_count


def process_collision_trajectory_from_one_pkl(pkl_path, algorithm, save_folder, feasibility_policy):
    if 'standard' not in algorithm:
        # don't process the standard data
        collision_vel_list = []
        collision_impulse_list = []
        collision_num = 0
        scenario_num = 0
        # collision_trajectory_list = []

        data = joblib.load(pkl_path)
        for sequence in tqdm(data.values()):
            scenario_num += 1
            CBV_vel, collision_impulse, collision_count = get_CBV_collision_severity(sequence)
            collision_vel_list.extend(CBV_vel)
            collision_impulse_list.extend(collision_impulse)
            collision_num += collision_count
            # collision_trajectory_list.append(get_collision_trajectory(sequence))

        collision_ratio = collision_num / scenario_num

        # # get overall infeasible_ratio
        # infeasible_ratio, feasibility_Vs = get_all_infeasible_ratio(collision_trajectory_list, feasibility_policy)
        # infeasible_distance, all_infeasible_distance = get_all_infeasible_distance(collision_trajectory_list, feasibility_policy)
        # print("infeasible distance", infeasible_distance)
        # print("infeasible_ratio", infeasible_ratio)

        # the overall information for collision trajectory
        collision_traj_info = {
            'collision_vel': np.array(collision_vel_list),
            'collision_impulse': np.array(collision_impulse_list),
            'collision_ratio': collision_ratio,
        }
        # save feasibility
        with open(osp.join(save_folder, "collision_traj_info.pkl"), 'wb') as pickle_file:
            pickle.dump(collision_traj_info, pickle_file)

