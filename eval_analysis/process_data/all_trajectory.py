#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：all_trajectory.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/4/22
"""
import os.path as osp
import pickle
import torch
from tqdm import tqdm
import joblib
from safebench.util.torch_util import CUDA, CPU


def get_all_feasibility(sequence):
    ego_obs = []
    for index, step in enumerate(sequence):
        ego_obs.append(step['ego_obs'])
    return ego_obs


def process_all_trajectory_from_one_pkl(pkl_path, feasibility_policy, save_folder):
    ego_obs_list = []

    data = joblib.load(pkl_path)
    for sequence in tqdm(data.values()):
        ego_obs = get_all_feasibility(sequence)
        ego_obs_list.extend(ego_obs)
    ego_obs_tensor = [torch.FloatTensor(ego_obs) for ego_obs in ego_obs_list]
    ego_obs_tensor = CUDA(torch.stack(ego_obs_tensor, dim=0))
    feasibility_Vs = feasibility_policy.get_feasibility_Vs(ego_obs_tensor)
    unfeasible_rate = (feasibility_Vs > 0).float().mean().item()

    # the overall information for all trajectories
    all_traj_info = {
        'feasibility_Vs': CPU(feasibility_Vs),
        'unfeasible_rate': unfeasible_rate,
    }
    # save feasibility
    with open(osp.join(save_folder, "all_traj_info.pkl"), 'wb') as pickle_file:
        pickle.dump(all_traj_info, pickle_file)
