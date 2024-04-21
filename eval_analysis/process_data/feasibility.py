#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：feasibility.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/4/18
"""
import os.path as osp
import pickle
import torch
from tqdm import tqdm
import joblib
from safebench.util.torch_util import CUDA, CPU


def process_feasibility_from_one_pkl(pkl_path, feasibility_policy, save_folder):

    ego_obs_list = []
    data = joblib.load(pkl_path)
    for sequence in tqdm(data.values()):
        for step in sequence:
            # only count the scenarios where the closest BV is CBV
            if step['ego_min_dis'] < 25 and step['BVs_id'][0] in step['CBVs_id']:
                ego_obs_list.append(torch.FloatTensor(step['ego_obs']))
    ego_obs_tensor = CUDA(torch.stack(ego_obs_list, dim=0))
    feasibility_Vs = feasibility_policy.get_feasibility_Vs(ego_obs_tensor)
    unfeasible_rate = (feasibility_Vs > 0).float().mean().item()
    feasibility = {'feasibility_Vs': CPU(feasibility_Vs), 'unfeasible_rate': unfeasible_rate}
    # save feasibility
    with open(osp.join(save_folder, "feasibility.pkl"), 'wb') as pickle_file:
        pickle.dump(feasibility, pickle_file)
    return feasibility_Vs
