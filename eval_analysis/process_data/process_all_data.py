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

from frea.util.run_util import load_config
from frea.feasibility import FEASIBILITY_LIST
from frea.util.logger import Logger, colorprint

from eval_analysis.process_data.all_trajectory import process_all_trajectory_from_one_pkl
from goal_reach_trajectory import process_goal_reach_trajectory_from_one_pkl
from collision_trajectory import process_collision_trajectory_from_one_pkl


def get_feasibility_policy(feasibility_config, algorithm, scenario_map, ROOT_DIR, eval_dir):
    logger_path = osp.join(ROOT_DIR, 'eval_analysis/processed_data', eval_dir, algorithm, scenario_map)
    logger = Logger(logger_path, scenario_map)
    feasibility_policy = FEASIBILITY_LIST[feasibility_config['type']](feasibility_config, logger=logger)
    map_name = scenario_map.split('_')[-1]
    feasibility_policy.load_model(map_name)
    return feasibility_policy


def list_directories(path):
    items = os.listdir(path)
    directories = [item for item in items if os.path.isdir(os.path.join(path, item))]
    return directories


def main(args_dict):
    ROOT_DIR = args_dict['ROOT_DIR']
    overload = args_dict['overload']
    feasibility_cfg_path = osp.join(ROOT_DIR, args_dict['feasibility_cfg_path'])
    feasibility_config = load_config(feasibility_cfg_path)
    feasibility_config.update(args_dict)

    eval_dirs = list_directories(osp.join(ROOT_DIR, 'log/eval'))
    for eval_dir in eval_dirs:
        colorprint('>> ' + '-' * 14 + f'Processing {eval_dir}' + '-' * 14, color='red')
        base_dir = os.path.join(osp.join(ROOT_DIR, 'log/eval', eval_dir))

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
                            print('>> ' + '-' * 70)
                            print(f'>> Reading {algorithm}, {scenario_map}')
                            pkl_path = osp.join(scenario_map_path, 'records.pkl')
                            save_folder = osp.join(ROOT_DIR, "eval_analysis", "processed_data", eval_dir, algorithm, scenario_map)
                            os.makedirs(save_folder, exist_ok=True)
                            if len(os.listdir(save_folder)) > 0 and not overload:
                                print('>> already have file, not overload')
                            else:
                                feasibility_policy = get_feasibility_policy(feasibility_config, algorithm, scenario_map, ROOT_DIR, eval_dir)
                                # collision trajectory for avoidable eval; miss trajectory for near-miss eval
                                if 'collision_traj' in args_dict['data']:
                                    print('>> Processing collision trajectory information')
                                    process_collision_trajectory_from_one_pkl(pkl_path, algorithm, save_folder, feasibility_policy)
                                if 'all_traj' in args_dict['data']:
                                    print('>> Processing all trajectory information')
                                    process_all_trajectory_from_one_pkl(pkl_path, algorithm, save_folder, feasibility_policy)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))))
    parser.add_argument('--data', '-d', nargs='*', type=str, default=['collision_traj', 'all_traj'])
    parser.add_argument('--overload', action='store_true')
    parser.add_argument('--feasibility_cfg_path', nargs='*', type=str, default='frea/feasibility/config/HJR.yaml')
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
