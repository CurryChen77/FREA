#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：process_PET.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/3/31
"""

import joblib
from scipy.spatial import cKDTree
from tqdm import tqdm
import os.path as osp
import os
import numpy as np


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def get_occupied_box_index_from_obs(x, y, x_list, y_list):
    x_id, near_x = find_nearest(x_list, x)
    y_id, near_y = find_nearest(y_list, y)
    surrounding_index = [
        [x_id, y_id],
        [x_id, y_id - 1], [x_id, y_id + 1],
        [x_id - 1, y_id], [x_id + 1, y_id],

    ]
    return surrounding_index


def calculate_pet_single_vehicle(ego_timestep, time_id_list):
    min_pet = 10
    for veh_time_id in time_id_list:
        time, veh_id = veh_time_id
        if veh_id == "ego":
            continue
        else:
            min_pet = min(min_pet, abs(float(time)-float(ego_timestep)))
    return min_pet


def calculate_position_pet_list(time_id_list):
    min_pet = 10
    if len(time_id_list) == 1:
        return min_pet
    for veh_time_id in time_id_list:
        time, veh_id = veh_time_id
        if veh_id == 'ego':
            min_pet_tmp = calculate_pet_single_vehicle(time, time_id_list)
            min_pet = min(min_pet, min_pet_tmp)
    return min_pet


def get_sequence_pet(sequence):
    pet_list = []
    pet_dict = {}

    x_max, x_min = max(sequence[0]['ego_x'], sequence[-1]['ego_x']), min(sequence[0]['ego_x'], sequence[-1]['ego_x'])
    y_max, y_min = max(sequence[0]['ego_y'], sequence[-1]['ego_y']), min(sequence[0]['ego_y'], sequence[-1]['ego_y'])

    x_list = np.linspace(x_min - 5, x_max + 5, num=(int(x_max - x_min)+10))
    y_list = np.linspace(y_min - 5, y_max + 5, num=(int(x_max - x_min)+10))

    for step in sequence:
        # add ego
        occupied_index_list = get_occupied_box_index_from_obs(step['ego_x'], step['ego_y'], x_list, y_list)
        for occupied_index in occupied_index_list:
            if str(occupied_index) in pet_dict:
                pet_dict[str(occupied_index)].append([step['current_game_time'], 'ego'])
            else:
                pet_dict[str(occupied_index)] = [[step['current_game_time'], 'ego']]
        # add all the bv
        for BV_index, BV_ego_dis in enumerate(step['BVs_ego_dis']):
            occupied_index_list = get_occupied_box_index_from_obs(
                step['BVs_abs_x'][BV_index], step['BVs_abs_y'][BV_index], x_list, y_list,
            )
            for occupied_index in occupied_index_list:
                if str(occupied_index) in pet_dict:
                    pet_dict[str(occupied_index)].append([step['current_game_time'], step['BVs_id'][BV_index]])
                else:
                    pet_dict[str(occupied_index)] = [[step['current_game_time'], step['BVs_id'][BV_index]]]

    for time_id_list in pet_dict.values():
        pet_tmp = calculate_position_pet_list(time_id_list)
        if pet_tmp < 10:
            pet_list.append(pet_tmp)

    return pet_list


def get_pet_list_from_one_pkl(pkl_path):
    pet_list_all_experiments = []
    data = joblib.load(pkl_path)
    for sequence in tqdm(data.values()):
        pet_list_all_experiments.extend(get_sequence_pet(sequence))
    return pet_list_all_experiments

