#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：feasibility.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/4/18
"""
import os.path as osp
import pickle
import torch
from tqdm import tqdm
import joblib
from safebench.util.torch_util import CUDA, CPU


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
                # reverse from the current step for each CBV ego collision event
                for i in range(index, -1, -1):
                    if CBV_id in sequence[i]['CBVs_id_set'] and CBV_id in sequence[i]['BVs_id_set']:
                        BV_index = sequence[i]['BVs_id'].index(CBV_id)
                        collision_trajectories[CBV_id].append({
                            'loc': sequence[i]['BVs_loc'][BV_index],
                            'vel': sequence[i]['BVs_vel'][BV_index],
                            'yaw': sequence[i]['BVs_yaw'][BV_index],
                            'time': sequence[i]['current_game_time'],
                            'ego_dis': sequence[i]['BVs_ego_dis'][BV_index]
                        })
                    else:
                        break
    return collision_trajectories


def get_CBV_unavoidable_collision(collision_trajectories):
    """
        analysis all the CBV collision trajectory and get whether it is avoidable
    """



def process_feasibility_from_one_pkl(pkl_path, feasibility_policy, save_folder):

    ego_obs_list = []
    data = joblib.load(pkl_path)
    for sequence in tqdm(data.values()):
        for step in sequence:
            # only count the scenarios where the closest BV is CBV
            if step['ego_min_dis'] < 25 and step['BVs_id'][0] in step['CBVs_id']:
                ego_obs_list.append(torch.FloatTensor(step['ego_obs']))
    ego_obs_tensor = CUDA(torch.stack(ego_obs_list, dim=0))
    feasibility_Vs = feasibility_policy.get_feasibility_Vs(ego_obs_tensor)
    unfeasible_rate = (feasibility_Vs > 0).float().mean().item()
    feasibility = {'feasibility_Vs': CPU(feasibility_Vs), 'unfeasible_rate': unfeasible_rate}
    # save feasibility
    with open(osp.join(save_folder, "feasibility.pkl"), 'wb') as pickle_file:
        pickle.dump(feasibility, pickle_file)
    return feasibility_Vs
