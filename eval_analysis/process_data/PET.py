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


def get_trajectory_pet(trajectory):
    """
        trajectory
        'ego': {'time': [], 'loc': []}
        'CBV_id': {'time': [], 'loc': [], 'ego_dis': []}
    """
    pet_list = []
    pet_dict = {}

    x_max, x_min = max(trajectory['ego']['loc'][0][0], trajectory['ego']['loc'][0][-1]), min(trajectory['ego']['loc'][0][0], trajectory['ego']['loc'][0][-1])
    y_max, y_min = max(trajectory['ego']['loc'][1][0], trajectory['ego']['loc'][1][-1]), min(trajectory['ego']['loc'][1][0], trajectory['ego']['loc'][1][-1])

    x_list = np.linspace(x_min - 5, x_max + 5, num=2*(int(x_max - x_min)+10))
    y_list = np.linspace(y_min - 5, y_max + 5, num=2*(int(x_max - x_min)+10))

    # add ego
    for i, ego_loc in enumerate(trajectory['ego']['loc']):
        occupied_index_list = get_occupied_box_index_from_obs(ego_loc, x_list, y_list)
        for occupied_index in occupied_index_list:
            if str(occupied_index) in pet_dict:
                pet_dict[str(occupied_index)].append([trajectory['ego']['time'][i], 'ego'])
            else:
                pet_dict[str(occupied_index)] = [[trajectory['ego']['time'][i], 'ego']]
    # got all the BV_id in the trajectory
    BV_ids = [key for key in trajectory if key != 'ego']
    # add each BV trajectory
    for BV_id in BV_ids:
        for i, BV_loc in enumerate(trajectory[BV_id]['loc']):
            occupied_index_list = get_occupied_box_index_from_obs(
                BV_loc, x_list, y_list,
            )
            for occupied_index in occupied_index_list:
                if str(occupied_index) in pet_dict:
                    pet_dict[str(occupied_index)].append([trajectory[BV_id]['time'][i], BV_id])
                else:
                    pet_dict[str(occupied_index)] = [[trajectory[BV_id]['time'][i], BV_id]]

    for time_id_list in pet_dict.values():
        pet_tmp = calculate_position_pet_list(time_id_list)
        if pet_tmp < 10:
            pet_list.append(pet_tmp)

    return pet_list

