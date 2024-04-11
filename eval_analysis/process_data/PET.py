#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：PET.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/3/31
"""

import joblib
from tqdm import tqdm
import os.path as osp
import numpy as np
import pickle


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def get_occupied_box_index_from_obs(loc, x_list, y_list):
    x, y = loc
    x_id, near_x = find_nearest(x_list, x)
    y_id, near_y = find_nearest(y_list, y)
    width_id_range = range(-4, 5)
    length_id_range = range(-4, 5)
    surrounding_index = [[x_id + dx, y_id + dy] for dy in width_id_range for dx in length_id_range]
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

    x_max, x_min = max(sequence[0]['ego_loc'][0], sequence[-1]['ego_loc'][0]), min(sequence[0]['ego_loc'][0], sequence[-1]['ego_loc'][0])
    y_max, y_min = max(sequence[0]['ego_loc'][1], sequence[-1]['ego_loc'][1]), min(sequence[0]['ego_loc'][1], sequence[-1]['ego_loc'][1])

    x_list = np.linspace(x_min - 5, x_max + 5, num=2*(int(x_max - x_min)+10))
    y_list = np.linspace(y_min - 5, y_max + 5, num=2*(int(x_max - x_min)+10))

    for step in sequence:
        # add ego
        occupied_index_list = get_occupied_box_index_from_obs(step['ego_loc'], x_list, y_list)
        for occupied_index in occupied_index_list:
            if str(occupied_index) in pet_dict:
                pet_dict[str(occupied_index)].append([step['current_game_time'], 'ego'])
            else:
                pet_dict[str(occupied_index)] = [[step['current_game_time'], 'ego']]
        # add all the bv
        for BV_index, BV_ego_dis in enumerate(step['BVs_ego_dis']):
            occupied_index_list = get_occupied_box_index_from_obs(
                step['BVs_loc'][BV_index], x_list, y_list,
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


def get_sequence_pet_avoidable(sequence):
    pet_list = []
    pet_dict = {}

    x_max, x_min = max(sequence[0]['ego_loc'][0], sequence[-1]['ego_loc'][0]), min(sequence[0]['ego_loc'][0], sequence[-1]['ego_loc'][0])
    y_max, y_min = max(sequence[0]['ego_loc'][1], sequence[-1]['ego_loc'][1]), min(sequence[0]['ego_loc'][1], sequence[-1]['ego_loc'][1])

    x_list = np.linspace(x_min - 5, x_max + 5, num=2*(int(x_max - x_min)+10))
    y_list = np.linspace(y_min - 5, y_max + 5, num=2*(int(x_max - x_min)+10))

    for step in sequence:
        if step['feasibility_V'] < 0:
            # add ego
            occupied_index_list = get_occupied_box_index_from_obs(step['ego_loc'], x_list, y_list)
            for occupied_index in occupied_index_list:
                if str(occupied_index) in pet_dict:
                    pet_dict[str(occupied_index)].append([step['current_game_time'], 'ego'])
                else:
                    pet_dict[str(occupied_index)] = [[step['current_game_time'], 'ego']]
            # add all the bv
            for BV_index, BV_ego_dis in enumerate(step['BVs_ego_dis']):
                occupied_index_list = get_occupied_box_index_from_obs(
                    step['BVs_loc'][BV_index], x_list, y_list,
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


def process_pet_from_one_pkl(pkl_path, save_folder):
    pet = {}
    pet_list_all = []
    pet_list_avoidable = []
    data = joblib.load(pkl_path)
    for sequence in tqdm(data.values()):
        pet_list_all.extend(get_sequence_pet(sequence))
        pet_list_avoidable.extend(get_sequence_pet_avoidable(sequence))
    # save the PET data
    pet['all_pet'] = pet_list_all
    pet['avoidable_pet'] = pet_list_avoidable
    # save Vehicle forward speed
    with open(osp.join(save_folder, "PET.pkl"), 'wb') as pickle_file:
        pickle.dump(pet, pickle_file)

