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
    logger_path = osp.join(ROOT_DIR, 'eval_analysis/King_analysis')
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
    all_infeasible_distance = []
    all_feasibility_Vs = []
    for all_fea_dis in collision_fea_dis.values():
        fea_obs = CUDA(torch.stack([torch.FloatTensor(row[0]) for row in all_fea_dis]))
        feasibility_Vs = feasibility_policy.get_feasibility_Vs(fea_obs)
        negative_index = find_negative_transition(feasibility_Vs)
        if negative_index:
            distance = torch.tensor([row[1] for row in all_fea_dis])
            all_infeasible_distance.append(distance[negative_index].item())
            all_feasibility_Vs.extend(CPU(feasibility_Vs))
    return all_infeasible_distance, all_feasibility_Vs


def get_collision_severity(all_sequence, all_file_ego_bv_dis_list, feasibility_policy):
    collision_fea_dis = {}
    for i, sequence in enumerate(all_sequence):
        collision_fea_dis[i] = []

        for step, ego_obs_state in enumerate(sequence):
            ego_BV_min_dis = all_file_ego_bv_dis_list[i][step]
            collision_fea_dis[i].append([ego_obs_state, ego_BV_min_dis])
    all_infeasible_distance, all_feasibility_Vs = get_feasibility_metric(collision_fea_dis, feasibility_policy)

    return all_infeasible_distance, all_feasibility_Vs


def get_final_closest_BV_index(sequences_data, num_agents):
    final_step_state = sequences_data[-1]
    sequences_length = len(sequences_data)
    min_dis = 25
    closest_BV = {}
    ego_coords, ego_yaw, ego_velocity = final_step_state['pos'][0], final_step_state['yaw'][0][0], final_step_state['vel'][0]
    for agent_index in range(1, num_agents + 1):
        vehicle_coords, vehicle_yaw, vehicle_velocity = final_step_state['pos'][agent_index], final_step_state['yaw'][agent_index][0], final_step_state['vel'][agent_index]
        ego_bv_dis = get_min_distance_across_boxes(ego_coords, vehicle_coords, ego_yaw, vehicle_yaw)
        if ego_bv_dis < min_dis:
            min_dis = ego_bv_dis
            closest_BV[agent_index] = [ego_bv_dis, sequences_length-1]

    return closest_BV


def get_closest_BV_indexes(sequences_data, num_agents):
    # init the BV index dict
    closest_BV = {}
    for i in range(1, num_agents + 1):
        closest_BV[i] = [25.0, 0]
    # find the closest step index for all BVs
    for step_index, step_state in enumerate(sequences_data):
        ego_coords = step_state['pos'][0]
        ego_yaw = step_state['yaw'][0][0]
        for agent_index in range(1, num_agents + 1):
            vehicle_coords = step_state['pos'][agent_index]
            vehicle_yaw = step_state['yaw'][agent_index][0]
            distance = get_min_distance_across_boxes(ego_coords, vehicle_coords, ego_yaw, vehicle_yaw)
            if distance < closest_BV[agent_index][0]:
                closest_BV[agent_index] = [distance, step_index]

    return closest_BV


def main(args_dict):
    ROOT_DIR = args_dict['ROOT_DIR']
    feasibility_cfg_path = osp.join(ROOT_DIR, args_dict['feasibility_cfg_path'])
    feasibility_config = load_config(feasibility_cfg_path)
    feasibility_config.update(args_dict)
    feasibility_policy = get_feasibility_policy(feasibility_config, ROOT_DIR)

    base_dir = osp.join(ROOT_DIR, "eval_analysis/King_analysis/data")
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
                        closest_BV = get_final_closest_BV_index(sequences_data, num_agents)
                        # closest_BV = get_closest_BV_indexes(sequences_data, num_agents)

                        for agent_index, data in closest_BV.items():
                            closest_dis, final_step_index = data

                            ego_obs_list = []
                            ego_bv_dis_list = []
                            for step_index in range(final_step_index, -1, -1):
                                step_state = sequences_data[step_index]
                                ego_info = [step_state['pos'][0], step_state['yaw'][0][0], step_state['vel'][0], [4.4, 1.8]]
                                vehicle_info = [step_state['pos'][agent_index], step_state['yaw'][agent_index][0], step_state['vel'][agent_index], [4.4, 1.8]]
                                ego_obs, ego_bv_dis = form_ego_obs(ego_info, vehicle_info)
                                ego_obs_list.append(ego_obs)
                                ego_bv_dis_list.append(ego_bv_dis)

                            all_file_ego_obs_list.append(ego_obs_list)
                            all_file_ego_bv_dis_list.append(ego_bv_dis_list)

    # get the collision severity of the King trajectories
    all_infeasible_dis, all_feasibility_Vs = get_collision_severity(all_file_ego_obs_list, all_file_ego_bv_dis_list, feasibility_policy)
    np_feasibility_Vs = np.array(all_feasibility_Vs)
    print('>> ' + '-' * 20, 'King analysis results', '-' * 20)
    print(">> Infeasible Distance:", round(np.mean(all_infeasible_dis), 2), "(m)")
    print(">> Infeasible Ratio:", round(np.mean(np_feasibility_Vs > 0) * 100, 2), "(%)")
    print(">> Collision Rate: ", round(collision_count/total_count * 100, 2), "(%)")
    print('>> ' + '-' * 20, 'King analysis results', '-' * 20)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))))
    parser.add_argument('--feasibility_cfg_path', nargs='*', type=str, default='frea/feasibility/config/HJR.yaml')
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)

