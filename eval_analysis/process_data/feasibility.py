#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：feasibility.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/5/9
"""
import torch

from frea.util.torch_util import CUDA, CPU


def get_trajectory_unfeasible_ratio(trajectory):
    ego_CBV_obs_list = []
    CBV_ids = [key for key in trajectory if key != 'ego']
    for CBV_id in CBV_ids:
        for i, ego_CBV_obs in enumerate(trajectory[CBV_id]['ego_CBV_obs']):
            ego_CBV_obs_list.append(torch.FloatTensor(ego_CBV_obs))
    return ego_CBV_obs_list


def get_overall_unfeasible_ratio(trajectories, feasibility_policy):
    ego_CBV_obs_list = []
    for trajectory in trajectories:
        ego_CBV_obs_list.extend(get_trajectory_unfeasible_ratio(trajectory))

    ego_obs_tensor = CUDA(torch.stack(ego_CBV_obs_list, dim=0))
    feasibility_Vs = feasibility_policy.get_feasibility_Vs(ego_obs_tensor)
    unfeasible_ratio = (feasibility_Vs > 0).float().mean().item()
    return unfeasible_ratio, CPU(feasibility_Vs),
