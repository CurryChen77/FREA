#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：plot_evaluation_result.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/2/28
"""

import pickle
import numpy as np
import seaborn as sns
import os.path as osp
import os
from collections import defaultdict
import re
import matplotlib
from matplotlib import pyplot as plt


def classified_data_by_ego(data):
    classified_by_ego = defaultdict(dict)

    for key, value in data.items():
        match = re.search(r"Ego:(\w+)", key)
        if match:
            ego = match.group(1)  # extract ego
            classified_by_ego[ego][key] = value

    classified_by_ego = dict(classified_by_ego)
    return classified_by_ego


def draw_data(All_data, data_name, ROOT_DIR, bins):
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    All_data_per_ego = classified_data_by_ego(All_data)
    for ego_type, datas in All_data_per_ego.items():
        num_algorithm = len(datas.keys())
        num_scenarios = max(len(scenarios) for scenarios in datas.values())
        color_list = sns.color_palette("flare", n_colors=num_algorithm-1)
        baseline_name = None
        subplots_height = 4
        aspect_ratio = 0.8
        num_cols = max(num_algorithm-1, 1)
        figsize = (subplots_height * aspect_ratio * num_cols, subplots_height)
        fig, axs = plt.subplots(nrows=num_scenarios, ncols=num_cols, figsize=figsize, squeeze=False)
        # plot the baseline
        for algorithm, scenario in datas.items():
            if algorithm.endswith('standard'):
                baseline_name = algorithm
                break
        # pop out the baseline data
        baseline = datas.pop(baseline_name, None) if baseline_name is not None else None
        for i in range(max(num_algorithm-1, 1)):
            for row, (scenario_name, data) in enumerate(baseline.items()):
                # sns.kdeplot(data, color=color_list[0], ax=axs[row, i], alpha=0.8, label=baseline_name, fill=True, linewidth=1)
                axs[row, i].hist(data, density=True, bins=bins, alpha=0.75, label=baseline_name, color=(144/255, 190/255, 224/255))
                axs[row, i].set_title(scenario_name, fontsize=10)
                axs[row, i].set_xlabel(f'{data_name}')
                axs[row, i].set_ylabel('Frequency')
                axs[row, i].legend(fontsize=7, loc='best')

        # plot the rest algorithm
        for i, (algorithm, scenario) in enumerate(datas.items()):
            for row, (scenario_name, data) in enumerate(scenario.items()):
                # sns.kdeplot(data, color=color_list[i+1], ax=axs[row, i], alpha=0.7, label=algorithm, fill=True, linewidth=1)
                axs[row, i].hist(data, density=True, bins=bins, alpha=0.6, label=algorithm, color=color_list[i])
                axs[row, i].legend(fontsize=7, loc='best')

        plt.tight_layout()
        data_save_name = data_name.replace(' ', "_")
        save_dir = osp.join(ROOT_DIR, f'eval_analysis/figures/{data_save_name}.png')
        plt.savefig(save_dir, dpi=600)
        plt.show()


def plot_metric(result, name):
    print('>> ' + '-' * 20, str(name), '-' * 20)
    for algorithm, scenario_data in result.items():
        for scenario_name, data in scenario_data.items():
            percentage = round(data * 100, 2)
            print(f">> {algorithm} in {scenario_name}: {percentage}")
    print('>> ' + '-' * 20, str(name), '-' * 20)


def main(args):
    ROOT_DIR = args.ROOT_DIR
    base_dir = osp.join(ROOT_DIR, 'eval_analysis/processed_data')
    algorithm_files = os.listdir(base_dir)
    PET_all_data = {}
    min_dis_data = {}
    near_rate = {}
    TTC_data = {}
    feasibility = {}
    unfeasible_rate = {}
    collision_ratio = {}
    collision_vel = {}
    collision_impulse = {}

    for algorithm in algorithm_files:
        if osp.isdir(osp.join(base_dir, algorithm)):
            split_name = algorithm.split('_')
            ego = split_name.pop(0)
            seed = split_name.pop(-1)
            select = split_name.pop(-1)
            cbv = '_'.join(split_name) if len(split_name) > 1 else split_name[0]
            algorithm_path = osp.join(base_dir, algorithm)
            if osp.isdir(algorithm_path):
                scenario_map_files = os.listdir(algorithm_path)
                # the specific ego and CBV method
                algorithm_title = f"Ego:{ego} CBV:{cbv}"

                PET_all_data[algorithm_title] = {}
                min_dis_data[algorithm_title] = {}
                near_rate[algorithm_title] = {}
                TTC_data[algorithm_title] = {}
                feasibility[algorithm_title] = {}
                unfeasible_rate[algorithm_title] = {}
                collision_ratio[algorithm_title] = {}
                collision_vel[algorithm_title] = {}
                collision_impulse[algorithm_title] = {}

                for scenario_map in scenario_map_files:
                    scenario_map_path = osp.join(algorithm_path, scenario_map)
                    if osp.isdir(scenario_map_path):
                        files = os.listdir(scenario_map_path)
                        for file in files:
                            # processing PET data
                            if file == 'collision_info.pkl' and 'collision_traj' in args.data:
                                file_path = os.path.join(scenario_map_path, file)
                                with open(file_path, 'rb') as pickle_file:
                                    all_collision_data = pickle.load(pickle_file)
                                collision_ratio[algorithm_title][scenario_map] = all_collision_data['collision_ratio']
                                collision_impulse[algorithm_title][scenario_map] = all_collision_data['collision_impulse']
                                feasibility[algorithm_title][scenario_map] = all_collision_data['feasibility_Vs']
                                unfeasible_rate[algorithm_title][scenario_map] = all_collision_data['unfeasible_rate']
                                collision_vel[algorithm_title][scenario_map] = all_collision_data['collision_vel']
                            if file == 'miss_info.pkl' and 'miss_traj' in args.data:
                                file_path = os.path.join(scenario_map_path, file)
                                with open(file_path, 'rb') as pickle_file:
                                    all_miss_data = pickle.load(pickle_file)

    if 'collision_traj' in args.data:
        plot_metric(unfeasible_rate, 'unfeasible_rate')
        plot_metric(collision_ratio, 'collision_ratio')

        feasibility_bins = np.linspace(-3, 5, 20)
        draw_data(feasibility, 'Feasibility Value', ROOT_DIR, bins=feasibility_bins)

        collision_impulse_bins = np.linspace(0, 20, 10)
        draw_data(collision_impulse, 'Collision impulse', ROOT_DIR, bins=collision_impulse_bins)

        collision_vel_bins = np.linspace(0, 10, 20)
        draw_data(collision_vel, 'Collision Velocity', ROOT_DIR, bins=collision_vel_bins)

    if 'miss_traj' in args.data:
        pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', '-m', action='store_true')
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))))
    parser.add_argument('--data', '-d', nargs='*', type=str, default=['collision_traj', 'miss_traj'])
    args = parser.parse_args()

    main(args)





