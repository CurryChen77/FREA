#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：TTC.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/4/2
"""
import joblib
from tqdm import tqdm
import os.path as osp
import numpy as np


def get_onestep_ttc(ego_loc, ego_vel, BV_loc, BV_vel):
    ego_x, ego_y = ego_loc
    ego_vx, ego_vy = ego_vel
    BV_x, BV_y = BV_loc
    BV_vx, BV_vy = BV_vel
    # Calculate the relative velocity vector
    v_rel_x = ego_vx - BV_vx
    v_rel_y = ego_vy - BV_vy

    # Calculate the position vector from BV to ego
    d_ab_x = ego_x - BV_x
    d_ab_y = ego_y - BV_y

    # Calculate the dot product of the position vector and the relative velocity vector
    dot_product = d_ab_x * v_rel_x + d_ab_y * v_rel_y

    # Calculate the square of the size of the relative velocity vector
    magnitude_squared = v_rel_x ** 2 + v_rel_y ** 2

    # Prevent division by zero
    if magnitude_squared == 0 and dot_product >= 0:
        return None

    # Calculate TTC
    ttc = -dot_product / magnitude_squared

    return ttc


def get_sequence_ttc(sequence):
    ttc_list = []

    for step in sequence:
        ego_loc = step['ego_loc']
        ego_vel = step['ego_vel']
        BVs_loc = step['BVs_loc']
        BVs_vel = step['BVs_vel']
        assert len(BVs_vel) == len(BVs_loc), 'length of BVs info should be the same'
        for i in range(len(BVs_vel)):
            ttc = get_onestep_ttc(ego_loc, ego_vel, BVs_loc[i], BVs_vel[i])
            ttc_list.append(ttc) if ttc is not None and ttc <= 5 else None

    return ttc_list


def process_ttc_from_one_pkl(pkl_path, save_folder):
    TTC_list_all_experiments = []
    data = joblib.load(pkl_path)
    for sequence in tqdm(data.values()):
        TTC_list_all_experiments.extend(get_sequence_ttc(sequence))
    # save the TTC data to npy
    np.save(osp.join(save_folder, "TTC.npy"), TTC_list_all_experiments)
