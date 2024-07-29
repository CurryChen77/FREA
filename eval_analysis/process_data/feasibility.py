#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：feasibility.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/5/9
"""
import numpy as np
import torch

from frea.util.torch_util import CUDA, CPU


def get_trajectory_infeasible_ratio(trajectory):
    ego_CBV_obs_list = []
    CBV_ids = [key for key in trajectory if key != 'ego']
    for CBV_id in CBV_ids:
        for i, ego_CBV_obs in enumerate(trajectory[CBV_id]['ego_CBV_obs']):
            ego_CBV_obs_list.append(torch.FloatTensor(ego_CBV_obs))
    return ego_CBV_obs_list


def get_all_infeasible_ratio(trajectories, feasibility_policy):
    ego_CBV_obs_list = []
    # for all route trajectory
    for trajectory in trajectories:
        # for all CBVs
        for CBV_id, data in trajectory.items():
            ego_CBV_obs_list.extend(data['ego_obs'])

    ego_obs_tensor = CUDA(torch.stack([torch.from_numpy(obs) for obs in ego_CBV_obs_list], dim=0))
    feasibility_Vs = feasibility_policy.get_feasibility_Vs(ego_obs_tensor)
    infeasible_ratio = (feasibility_Vs > 0).float().mean().item()
    return infeasible_ratio, CPU(feasibility_Vs),


def find_negative_transition(fea_V):
    # find the negative situation
    mask = (fea_V < 0) & (torch.roll(fea_V, shifts=-1) < 0)
    # find the first index
    start_index = torch.argmax(mask.int()).item()
    return start_index if mask.any() else None


def get_infeasible_distance(data, feasibility_policy):
    ego_obs = data['ego_obs']
    ego_cbv_dis = data['ego_cbv_dis']
    fea_obs = CUDA(torch.stack([torch.FloatTensor(obs) for obs in ego_obs]))
    feasibility_Vs = feasibility_policy.get_feasibility_Vs(fea_obs)
    negative_index = find_negative_transition(feasibility_Vs)
    if negative_index:
        distance = torch.tensor([dis for dis in ego_cbv_dis])
        return distance[negative_index].item()
    else:
        return None


def get_all_infeasible_distance(trajectories, feasibility_policy):
    infeasible_distance_list = []
    for trajectory in trajectories:
        for CBV_id, data in trajectory.items():
            infeasible_distance = get_infeasible_distance(data, feasibility_policy)
            if infeasible_distance:
                infeasible_distance_list.append(infeasible_distance)

    infeasible_distance_list = np.array(infeasible_distance_list)
    mean_infeasible_distance = np.mean(infeasible_distance_list)
    return mean_infeasible_distance, infeasible_distance_list
