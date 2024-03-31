#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：process_PET.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/3/31
"""

import joblib
from scipy.spatial import cKDTree
from tqdm import tqdm
import os.path as osp
import os
import numpy as np


def get_occupied_box_index_from_obs(x, y, tree):

    min_distance, min_distance_index = tree.query([x, y])

    index = min_distance_index if min_distance < 4 else None

    return index


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
    ego_time, ego_id = time_id_list[0]
    assert ego_id == 'ego', 'ego should at the first line'
    min_pet_tmp = calculate_pet_single_vehicle(ego_time, time_id_list)
    min_pet = min(min_pet, min_pet_tmp)

    return min_pet


def get_sequence_pet(sequence):
    ego_loc = []
    pet_list = []
    pet_dict = {}
    index = 0
    for i, step in enumerate(sequence):
        if i % 2 == 0:
            ego_loc.append([step['ego_x'], step['ego_y']])
            pet_dict[str(index)] = [[step['current_game_time'], 'ego']]
            index += 1
    tree = cKDTree(ego_loc)
    for step in sequence:
        for BV_index, BV_ego_dis in enumerate(step['BVs_ego_dis']):
            if BV_ego_dis < 15:
                index = get_occupied_box_index_from_obs(step['BVs_abs_x'][BV_index], step['BVs_abs_y'][BV_index], tree)
                if index is not None:
                    pet_dict[str(index)].append([step['current_game_time'], step['BVs_id'][BV_index]])

    for time_id_list in pet_dict.values():
        pet_tmp = calculate_position_pet_list(time_id_list)
        if pet_tmp < 10:
            pet_list.append(pet_tmp)

    return pet_list


def get_pet_list_from_one_pkl(pkl_path):
    pet_list_all_experiments = []
    data = joblib.load(pkl_path)
    for sequence in tqdm(data.values()):
        pet_list_all_experiments.extend(get_sequence_pet(sequence))
    return pet_list_all_experiments


def main(ROOT_DIR, args):
    algorithm_files = os.listdir(ROOT_DIR)
    algorithm_titles = []
    for algorithm in algorithm_files:
        split_name = algorithm.split('_')
        ego = split_name.pop(0)
        seed = split_name.pop(-1)
        select = split_name.pop(-1)
        cbv = '_'.join(split_name) if len(split_name) > 1 else split_name[0]
        algorithm_path = osp.join(ROOT_DIR, algorithm)
        if osp.isdir(algorithm_path):
            scenario_map_files = os.listdir(algorithm_path)
            velocity = {name: [] for name in scenario_map_files}
            ego_dis = {name: [] for name in scenario_map_files}
            algorithm_title = f"Ego:{ego} CBV:{cbv}"
            algorithm_titles.append(algorithm_title)

            for scenario_map in scenario_map_files:
                scenario_map_path = osp.join(algorithm_path, scenario_map)
                scenario_map_title = f'\n{scenario_map}'
                if osp.isdir(scenario_map_path):
                    files = os.listdir(scenario_map_path)
                    if 'records.pkl' in files:
                        pkl_path = osp.join(scenario_map_path, 'records.pkl')
                        # get PET
                        pet_list_all_experiments = get_pet_list_from_one_pkl(pkl_path)
                        save_folder = osp.join(args.ROOT_DIR, "eval_analysis", "processed_data", algorithm, scenario_map)
                        os.makedirs(save_folder, exist_ok=True)
                        np.save(osp.join(save_folder, "PET.npy"), pet_list_all_experiments)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))))
    args = parser.parse_args()

    ROOT_DIR = osp.join(args.ROOT_DIR, 'log/eval')

    main(ROOT_DIR, args)
