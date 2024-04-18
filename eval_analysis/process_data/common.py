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
import torch
from collections import Counter
from tqdm import tqdm
import joblib
import numpy as np
import math
from safebench.scenario.scenario_definition.atomic_criteria import Status
from safebench.util.torch_util import CUDA, CPU


def process_feasibility_from_one_pkl(pkl_path, feasibility_policy, save_folder):

    ego_obs_list = []
    data = joblib.load(pkl_path)
    for sequence in tqdm(data.values()):
        for step in sequence:
            if step['ego_min_dis'] < 25:
                ego_obs_list.append(torch.FloatTensor(step['ego_obs']))
    ego_obs_tensor = CUDA(torch.stack(ego_obs_list, dim=0))
    feasibility_Vs = feasibility_policy.get_feasibility_Vs(ego_obs_tensor)
    unfeasible_rate = (feasibility_Vs > 0).float().mean().item()
    feasibility = {'feasibility_Vs': CPU(feasibility_Vs), 'unfeasible_rate': unfeasible_rate}
    # save feasibility
    with open(osp.join(save_folder, "feasibility.pkl"), 'wb') as pickle_file:
        pickle.dump(feasibility, pickle_file)
    return feasibility_Vs


def process_collision_from_one_pkl(pkl_path, algorithm, save_folder):
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
        collision_attacker = 0
        for sequence in tqdm(data.values()):
            collision_count = Counter()
            all_count = Counter()

            for step in sequence:
                for CBV_id, collision_event in step['CBVs_collision'].items():
                    if collision_event is not None:
                        collision_count[CBV_id] += 1
                        step_collision_impulse = collision_event['normal_impulse']
                        collision_impulse.append(np.sqrt(np.sum(np.square(np.array(step_collision_impulse) / 1000))))  # kN*s
                    all_count[CBV_id] += 1

            num_collision += len(collision_count)
            collision_attacker += len(all_count)

        collision['collision_rate'] = num_collision / collision_attacker
        collision['collision_impulse'] = collision_impulse
    # save Vehicle forward speed
    with open(osp.join(save_folder, "collision.pkl"), 'wb') as pickle_file:
        pickle.dump(collision, pickle_file)


def process_common_data_from_one_pkl(pkl_path, save_folder):
    total_step = 0
    near_count = 0
    Vehicle_forward_speed = []
    min_dis = []
    data = joblib.load(pkl_path)
    for sequence in tqdm(data.values()):
        for step in sequence:
            if step['ego_min_dis'] < 25:
                total_step += 1

                min_dis.append(step['ego_min_dis'])
                near_count += 1 if step['ego_min_dis'] < 1 else 0

                for vel in step['BVs_vel']:
                    abs_vel = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
                    Vehicle_forward_speed.append(abs_vel) if abs_vel > 0.1 else None

    min_dis_data = {
        'near_rate': near_count / total_step,
        'min_dis': min_dis
    }
    # save ego min dis
    with open(osp.join(save_folder, "min_dis.pkl"), 'wb') as pickle_file:
        pickle.dump(min_dis_data, pickle_file)
    # save Vehicle forward speed
    np.save(osp.join(save_folder, "vehicle_forward_speed.npy"), Vehicle_forward_speed)


