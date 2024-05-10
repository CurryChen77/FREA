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
from frea.util.torch_util import CUDA, CPU


def find_negative_transition(fea_V):
    # find the negative situation
    mask = (fea_V < 0) & (torch.roll(fea_V, shifts=-1) < 0)
    # find the first index
    start_index = torch.argmax(mask.int()).item()
    return start_index if mask.any() else None


def get_feasibility_boundary_dis(collision_fea_dis, feasibility_policy):
    fea_boundary_dis = []
    for fea_dis in collision_fea_dis.values():
        fea_obs = CUDA(torch.stack([torch.FloatTensor(row[0]) for row in fea_dis]))
        feasibility_Vs = feasibility_policy.get_feasibility_Vs(fea_obs)
        negative_index = find_negative_transition(feasibility_Vs)
        if negative_index:
            distance = torch.tensor([row[1] for row in fea_dis])
            fea_boundary_dis.append(distance[negative_index].item())
    return fea_boundary_dis


def get_CBV_collision_severity(sequence, feasibility_policy):
    CBV_vel = []
    collision_count = 0
    collision_impulse = []
    collision_fea_dis = {}
    for index, step in enumerate(sequence):
        for CBV_id, collision_event in step['CBVs_collision'].items():
            # only count the CBV collision with the ego vehicle
            if collision_event is not None and collision_event['other_actor_id'] == step['ego_id']:
                # collision impulse
                collision_impulse.append(np.sqrt(np.sum(np.square(np.array(collision_event['normal_impulse']) / 1000))))  # kN*s
                collision_count += 1

                collision_fea_dis[CBV_id] = []
                for i in range(index, -1, -1):
                    # reverse the whole trajectory of the collision CBV
                    if CBV_id in sequence[i]['CBVs_id'] and CBV_id in sequence[i]['BVs_id']:
                        BV_index = sequence[i]['BVs_id'].index(CBV_id)
                        # record the velocity
                        CBV_vel.append(np.linalg.norm(np.array(sequence[i]['BVs_vel'][BV_index])))
                        collision_fea_dis[CBV_id].append([sequence[i]['ego_CBV_obs'][CBV_id], sequence[i]['BVs_ego_dis'][BV_index]])
                    else:
                        break

    fea_boundary_dis = get_feasibility_boundary_dis(collision_fea_dis, feasibility_policy)

    return CBV_vel, collision_impulse, collision_count, fea_boundary_dis


def process_collision_trajectory_from_one_pkl(pkl_path, save_folder, feasibility_policy):
    collision_vel_list = []
    collision_impulse_list = []
    fea_boundary_dis_list = []
    collision_num = 0
    scenario_num = 0

    data = joblib.load(pkl_path)
    for sequence in tqdm(data.values()):
        scenario_num += 1
        CBV_vel, collision_impulse, collision_count, fea_boundary_dis = get_CBV_collision_severity(sequence, feasibility_policy)
        collision_vel_list.extend(CBV_vel)
        collision_impulse_list.extend(collision_impulse)
        collision_num += collision_count
        fea_boundary_dis_list.extend(fea_boundary_dis)

    fea_boundary_dis_list = np.array(fea_boundary_dis_list)

    collision_ratio = collision_num / scenario_num

    # the overall information for collision trajectory
    collision_traj_info = {
        'collision_vel': np.array(collision_vel_list),
        'collision_impulse': np.array(collision_impulse_list),
        'collision_ratio': collision_ratio,
        'fea_boundary_dis': fea_boundary_dis_list,
        'fea_boundary_dis_mean': np.mean(fea_boundary_dis_list) if len(fea_boundary_dis_list) > 0 else 100
    }
    # save feasibility
    with open(osp.join(save_folder, "collision_traj_info.pkl"), 'wb') as pickle_file:
        pickle.dump(collision_traj_info, pickle_file)

