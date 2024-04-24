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


def rotate_point(cx, cy, angle, px, py):
    # rotate (px, py) from (cx, cy) with an angle
    s = np.sin(angle)
    c = np.cos(angle)
    px -= cx
    py -= cy
    xnew = px * c - py * s
    ynew = px * s + py * c
    px = xnew + cx
    py = ynew + cy
    return px, py


def get_all_grid_indices_within_corners(corner_indices):
    # all corners indices
    x_indices, y_indices = zip(*corner_indices)

    # find the min and max indices
    min_x, max_x = min(x_indices), max(x_indices)
    min_y, max_y = min(y_indices), max(y_indices)

    if min_x <= max_x < min_x + 1:
        x_range = range(min_x, max_x)
    else:
        x_range = range(min_x + 1, max_x)

    if min_y <= max_y < min_y + 1:
        y_range = range(min_y, max_y)
    else:
        y_range = range(min_y + 1, max_y)

    # get all the combination of the x, y indices with in the range
    occupied_indices = [(xi, yi) for xi in x_range for yi in y_range]

    return occupied_indices


def get_occupied_box_index_from_obs(loc, x_list, y_list, extent, yaw):
    x, y = loc
    extent_x = extent[0] / 2
    extent_y = extent[1] / 2

    corners = [
        (x - extent_x, y - extent_y),
        (x - extent_x, y + extent_y),
        (x + extent_x, y - extent_y),
        (x + extent_x, y + extent_y),
    ]
    # get the rotated corners positions
    rotated_corners = [rotate_point(x, y, yaw, cx, cy) for cx, cy in corners]
    # get the rotated corners indices
    corner_indices = [[find_nearest(x_list, cx)[0], find_nearest(y_list, cy)[0]] for cx, cy in rotated_corners]
    # get all the indices with in the rotated corners
    surrounding_index = get_all_grid_indices_within_corners(corner_indices)
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
    x_min = min(trajectory['ego']['loc'], key=lambda item: item[0])[0]
    x_max = max(trajectory['ego']['loc'], key=lambda item: item[0])[0]
    y_min = min(trajectory['ego']['loc'], key=lambda item: item[1])[1]
    y_max = max(trajectory['ego']['loc'], key=lambda item: item[1])[1]

    step = 0.5
    x_list = np.arange(x_min - 7, x_max + 7 + step, step)
    y_list = np.arange(y_min - 7, y_max + 7 + step, step)

    # add ego
    for i, ego_loc in enumerate(trajectory['ego']['loc']):
        occupied_index_list = get_occupied_box_index_from_obs(
            ego_loc, x_list, y_list, trajectory['ego']['extent'][i], trajectory['ego']['yaw'][i]
        )
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
                BV_loc, x_list, y_list, trajectory[BV_id]['extent'][i], trajectory['ego']['yaw'][i]
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

