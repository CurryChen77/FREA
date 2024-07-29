#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：King_analysis.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/6/12
"""
import os
import os.path as osp
import json
import numpy as np
import torch
from tqdm import tqdm

from eval_analysis.process_data.all_trajectory import get_min_distance_across_centers, get_relative_info, get_min_distance_across_boxes, form_ego_obs
from frea.util.torch_util import CUDA, CPU
from frea.feasibility import FEASIBILITY_LIST
from frea.util.logger import Logger
from frea.util.run_util import load_config


def get_feasibility_policy(feasibility_config, ROOT_DIR):
    logger_path = osp.join(ROOT_DIR, 'eval_analysis/King_result')
    logger = Logger(logger_path, all_map_name=['Town05, Town02'])
    feasibility_policy = FEASIBILITY_LIST[feasibility_config['type']](feasibility_config, logger=logger)
    feasibility_policy.load_model(map_name='Town05')
    print(">> Successfully loading the feasibility policy")
    return feasibility_policy


def sort_vehicle_infos_by_distance(ego_info, vehicle_infos):
    ego_coords, ego_yaw, ego_velocity = ego_info
    sorted_vehicle_infos = sorted(
        vehicle_infos,
        key=lambda vehicle_info: get_min_distance_across_centers(np.array(ego_coords), np.array(vehicle_info[0]))
    )

    return sorted_vehicle_infos


def find_negative_transition(fea_V):
    # find the negative situation
    mask = (fea_V < 0) & (torch.roll(fea_V, shifts=-1) < 0)
    # find the first index
    start_index = torch.argmax(mask.int()).item()
    return start_index if mask.any() else None


def get_feasibility_metric(collision_fea_dis, feasibility_policy):
    fea_boundary_dis = []
    feasibility_Vs = None
    for all_fea_dis in collision_fea_dis.values():
        fea_obs = CUDA(torch.stack([torch.FloatTensor(row[0]) for row in all_fea_dis]))
        feasibility_Vs = feasibility_policy.get_feasibility_Vs(fea_obs)
        reversed_fea_V = torch.flip(feasibility_Vs, dims=[0])
        negative_index = find_negative_transition(reversed_fea_V)
        if negative_index:
            distance = torch.tensor([row[1] for row in all_fea_dis])
            reversed_distance = torch.flip(distance, dims=[0])
            fea_boundary_dis.append(reversed_distance[negative_index].item())
    return fea_boundary_dis, feasibility_Vs


def get_collision_severity(all_sequence, all_file_ego_bv_dis_list, feasibility_policy):
    collision_fea_dis = {}
    for i, sequence in enumerate(all_sequence):
        collision_fea_dis[i] = []

        for step, ego_obs_state in enumerate(sequence):
            ego_BV_min_dis = all_file_ego_bv_dis_list[i][step]
            collision_fea_dis[i].append([ego_obs_state, ego_BV_min_dis])

    fea_boundary_dis, feasibility_Vs = get_feasibility_metric(collision_fea_dis, feasibility_policy)

    return fea_boundary_dis, feasibility_Vs


def get_final_closest_BV_index(final_step_state, num_agents):
    min_dis = 25
    BV_index = None
    ego_coords, ego_yaw, ego_velocity = final_step_state['pos'][0], final_step_state['yaw'][0][0], final_step_state['vel'][0]
    for agent_index in range(1, num_agents + 1):
        vehicle_coords, vehicle_yaw, vehicle_velocity = final_step_state['pos'][agent_index], final_step_state['yaw'][agent_index][0], final_step_state['vel'][agent_index]
        ego_bv_dis = get_min_distance_across_boxes(ego_coords, vehicle_coords, ego_yaw, vehicle_yaw)
        if ego_bv_dis < min_dis:
            min_dis = ego_bv_dis
            BV_index = agent_index

    return BV_index


def main(args_dict):
    ROOT_DIR = args_dict['ROOT_DIR']
    feasibility_cfg_path = osp.join(ROOT_DIR, args_dict['feasibility_cfg_path'])
    feasibility_config = load_config(feasibility_cfg_path)
    feasibility_config.update(args_dict)
    feasibility_policy = get_feasibility_policy(feasibility_config, ROOT_DIR)

    base_dir = "eval_analysis/King_analysis/data"
    collision_count = 0
    total_count = 0
    all_file_ego_obs_list = []
    all_file_ego_bv_dis_list = []

    for folder in os.listdir(base_dir):
        print(f"Processing {folder}")
        folder_path = os.path.join(base_dir, folder)
        for agent_folder in tqdm(os.listdir(folder_path)):
            agent_folder_path = os.path.join(folder_path, agent_folder)
            if os.path.isdir(agent_folder_path):
                record_file_path = os.path.join(agent_folder_path, 'scenario_records.json')
                result_file_path = os.path.join(agent_folder_path, 'results.json')

                if os.path.exists(record_file_path) and os.path.exists(result_file_path):
                    with open(result_file_path, 'r') as file:
                        result_data = json.load(file)
                        # only use the final iteration's result
                        iteration = result_data["first_metrics"]["iteration"]
                        town = result_data["meta_data"]["town"]
                        num_agents = result_data["meta_data"]["Num_agents"]

                    with open(record_file_path, 'r') as file:
                        record_data = json.load(file)
                    if town == 'Town05':
                        total_count += 1
                        # only evaluate the collision trajectory
                        collision = result_data["all_iterations"][str(iteration)]["Collision Metric"]
                        # record the collision count
                        if collision == 1.0:
                            collision_count += 1

                        # read the best iteration data
                        sequences_data = record_data["states"][iteration]

                        # find the closest BV index at the final step
                        agent_index = get_final_closest_BV_index(sequences_data[-1], num_agents)

                        if agent_index:
                            ego_obs_list = []
                            ego_bv_dis_list = []
                            for i, step_state in enumerate(sequences_data):
                                # only using the closest BV to calculate infeasible distance and infeasible ratio
                                ego_info = [step_state['pos'][0], step_state['yaw'][0][0], step_state['vel'][0], [4.4, 1.8]]
                                vehicle_info = [step_state['pos'][agent_index], step_state['yaw'][agent_index][0], step_state['vel'][agent_index], [4.4, 1.8]]
                                ego_obs, ego_bv_dis = form_ego_obs(ego_info, vehicle_info)
                                ego_obs_list.append(ego_obs)
                                ego_bv_dis_list.append(ego_bv_dis)

                            all_file_ego_obs_list.append(ego_obs_list)
                            all_file_ego_bv_dis_list.append(ego_bv_dis_list)

    # get the collision severity of the King trajectories
    all_infeasible_dis, feasibility_Vs = get_collision_severity(all_file_ego_obs_list, all_file_ego_bv_dis_list, feasibility_policy)
    print("Infeasible Distance (m):", round(np.mean(all_infeasible_dis), 2))
    print("Infeasible Ratio (%) :", round((feasibility_Vs > 0).float().mean().item() * 100, 2))
    print(f"Collision Rate (%): ", round(collision_count/total_count * 100, 2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))))
    parser.add_argument('--feasibility_cfg_path', nargs='*', type=str, default='frea/feasibility/config/HJR.yaml')
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)

