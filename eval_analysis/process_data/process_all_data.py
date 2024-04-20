#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：process_all_data.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/4/1
"""
import os
import os.path as osp
import numpy as np

from safebench.util.run_util import load_config
from safebench.feasibility import FEASIBILITY_LIST
from safebench.util.logger import Logger
from common import process_common_data_from_one_pkl, process_collision_from_one_pkl
from feasibility import process_feasibility_from_one_pkl
from PET import process_pet_from_one_pkl
from eval_analysis.process_data.TTC import process_ttc_from_one_pkl


def get_feasibility_policy(feasibility_config, algorithm, scenario_map, ROOT_DIR):
    logger_path = osp.join(ROOT_DIR, 'eval_analysis/processed_data', algorithm, scenario_map)
    logger = Logger(logger_path, scenario_map)
    feasibility_policy = FEASIBILITY_LIST[feasibility_config['type']](feasibility_config, logger=logger)
    map_name = scenario_map.split('_')[-1]
    feasibility_policy.load_model(map_name)
    return feasibility_policy


def main(args_dict):
    ROOT_DIR = args_dict['ROOT_DIR']
    base_dir = osp.join(ROOT_DIR, 'log/eval')
    feasibility_cfg_path = osp.join(ROOT_DIR, args_dict['feasibility_cfg_path'])
    feasibility_config = load_config(feasibility_cfg_path)
    feasibility_config.update(args_dict)

    algorithm_files = os.listdir(base_dir)
    for algorithm in algorithm_files:
        algorithm_path = osp.join(base_dir, algorithm)
        if osp.isdir(algorithm_path):
            scenario_map_files = os.listdir(algorithm_path)
            for scenario_map in scenario_map_files:
                scenario_map_path = osp.join(algorithm_path, scenario_map)
                if osp.isdir(scenario_map_path):
                    files = os.listdir(scenario_map_path)
                    if 'records.pkl' in files:
                        print('>> ' + '-' * 40)
                        print(f'>> Reading {algorithm}, {scenario_map}')
                        pkl_path = osp.join(scenario_map_path, 'records.pkl')
                        save_folder = osp.join(ROOT_DIR, "eval_analysis", "processed_data", algorithm, scenario_map)
                        os.makedirs(save_folder, exist_ok=True)
                        feasibility_policy = get_feasibility_policy(feasibility_config, algorithm, scenario_map, ROOT_DIR)
                        if 'feasibility' in args_dict['data']:
                            print('>> Processing Feasibility')
                            feasibility_Vs = process_feasibility_from_one_pkl(pkl_path, feasibility_policy, save_folder)
                        if 'PET' in args_dict['data']:
                            print('>> Processing PET')
                            process_pet_from_one_pkl(pkl_path, save_folder)
                        if 'common' in args_dict['data']:
                            print('>> Processing common data')
                            process_common_data_from_one_pkl(pkl_path, save_folder)
                        if 'TTC' in args_dict['data']:
                            print('>> Processing TTC data')
                            process_ttc_from_one_pkl(pkl_path, save_folder)
                        if 'collision' in args_dict['data']:
                            print('>> Processing Collision data')
                            process_collision_from_one_pkl(pkl_path, algorithm, save_folder)
                        print('>> ' + '-' * 40)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))))
    parser.add_argument('--data', '-d', nargs='*', type=str, default=['common', 'PET', 'TTC', 'collision', 'feasibility'])
    parser.add_argument('--feasibility_cfg_path', nargs='*', type=str, default='safebench/feasibility/config/HJR.yaml')
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
