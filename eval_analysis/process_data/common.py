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
                    if collision_event is not None and collision_event['other_actor_id'] == step['ego_id']:
                        # only count the collision with ego vehicle
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


