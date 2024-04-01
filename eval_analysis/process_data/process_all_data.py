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

from precess_common_state import process_common_data_from_one_pkl
from process_PET import process_pet_from_one_pkl


def main(args):
    base_dir = osp.join(args.ROOT_DIR, 'log/eval')
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
                        print('>> Processing PET')
                        # get PET
                        save_folder = osp.join(args.ROOT_DIR, "eval_analysis", "processed_data", algorithm, scenario_map)
                        os.makedirs(save_folder, exist_ok=True)
                        process_pet_from_one_pkl(pkl_path, save_folder)
                        print('>> Processing common data')
                        process_common_data_from_one_pkl(pkl_path, save_folder)
                        print('>> ' + '-' * 40)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))))
    args = parser.parse_args()

    main(args)
