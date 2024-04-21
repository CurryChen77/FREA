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
import torch
from tqdm import tqdm
import joblib
from safebench.util.torch_util import CUDA, CPU


def get_CBV_collision_feasibility(sequence):
    ego_obs = []
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
                for i in range(index, -1, -1):
                    # count the time the CBV is the closest BV from ego
                    if CBV_id in sequence[i]['CBVs_id'] and CBV_id == sequence[i]['BVs_id'][0] and sequence[i]['BVs_ego_dis'][0] < 15:
                        BV_index = sequence[i]['BVs_id'].index(CBV_id)
                        # record the velocity and the ego obs
                        CBV_vel.append(np.linalg.norm(np.array(sequence[i]['BVs_vel'][BV_index])))
                        ego_obs.append(sequence[i]['ego_obs'])
                    else:
                        break
    return ego_obs, CBV_vel, collision_impulse, collision_count


def process_collision_trajectory_from_one_pkl(pkl_path, feasibility_policy, save_folder):
    ego_obs_list = []
    collision_vel_list = []
    collision_impulse_list = []
    collision_num = 0
    scenario_num = 0

    data = joblib.load(pkl_path)
    for sequence in tqdm(data.values()):
        scenario_num += 1
        ego_obs, CBV_vel, collision_impulse, collision_count = get_CBV_collision_feasibility(sequence)
        ego_obs_list.extend(torch.FloatTensor(ego_obs))
        collision_vel_list.extend(CBV_vel)
        collision_impulse_list.extend(collision_impulse)
        collision_num += collision_count

    ego_obs_tensor = CUDA(torch.stack(ego_obs_list, dim=0))
    feasibility_Vs = feasibility_policy.get_feasibility_Vs(ego_obs_tensor)
    unfeasible_rate = (feasibility_Vs > 0).float().mean().item()
    collision_ratio = collision_num / scenario_num

    # the overall information for collision trajectory
    collision_info = {
        'feasibility_Vs': CPU(feasibility_Vs),
        'unfeasible_rate': unfeasible_rate,
        'collision_vel': np.array(collision_vel_list),
        'collision_impulse': np.array(collision_impulse_list),
        'collision_ratio': collision_ratio
    }
    # save feasibility
    with open(osp.join(save_folder, "collision_info.pkl"), 'wb') as pickle_file:
        pickle.dump(collision_info, pickle_file)

