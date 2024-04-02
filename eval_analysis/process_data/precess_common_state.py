#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：process_common_state.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/4/1
"""
import os
import os.path as osp
from tqdm import tqdm
import joblib
import numpy as np


def process_common_data_from_one_pkl(pkl_path, save_folder):
    ego_min_dis_list = []
    BVs_forward_speed = []
    data = joblib.load(pkl_path)
    for sequence in tqdm(data.values()):
        for step in sequence:
            ego_min_dis_list.append(step['ego_min_dis']) if step['ego_min_dis'] < 25 else None
            for forward_speed in step['BVs_forward_speed']:
                BVs_forward_speed.append(forward_speed) if forward_speed > 0.1 else None
    # save ego min dis
    np.save(osp.join(save_folder, "Ego_min_dis.npy"), ego_min_dis_list)
    # save BVs forward speed
    np.save(osp.join(save_folder, "BVs_forward_speed.npy"), BVs_forward_speed)

