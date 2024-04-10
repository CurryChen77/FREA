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
import math
from safebench.scenario.scenario_definition.atomic_criteria import Status


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
    # save BVs forward speed
    with open(osp.join(save_folder, "Collision.pkl"), 'wb') as pickle_file:
        pickle.dump(collision, pickle_file)


def process_common_data_from_one_pkl(pkl_path, save_folder):
    total_step = 0
    unfeasible_count = 0
    near_count = 0
    BVs_forward_speed = []
    ego_min_dis = {'ego_min_dis': []}
    feasibility = {'feasibility_value': []}
    data = joblib.load(pkl_path)
    for sequence in tqdm(data.values()):
        for step in sequence:
            if step['ego_min_dis'] < 25:
                total_step += 1
                ego_min_dis['ego_min_dis'].append(step['ego_min_dis'])
                if step['ego_min_dis'] < 1:
                    near_count += 1
                for vel in step['BVs_vel']:
                    abs_vel = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
                    BVs_forward_speed.append(abs_vel) if abs_vel > 0.1 else None
                if 'feasibility_V' in step:
                    feasibility['feasibility_value'].append(step['feasibility_V'])
                    if step['feasibility_V'] > 0:
                        unfeasible_count += 1

    feasibility['unfeasible_rate'] = unfeasible_count / total_step
    ego_min_dis['near_rate'] = near_count / total_step

    # save ego min dis
    with open(osp.join(save_folder, "Ego_min_dis.pkl"), 'wb') as pickle_file:
        pickle.dump(ego_min_dis, pickle_file)
    # save BVs forward speed
    np.save(osp.join(save_folder, "BVs_forward_speed.npy"), BVs_forward_speed)
    # save feasibility
    with open(osp.join(save_folder, "Feasibility.pkl"), 'wb') as pickle_file:
        pickle.dump(feasibility, pickle_file)

