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


def draw_data(All_data, data_name, ROOT_DIR, bins, baseline_CBV='standard', density=True):
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
            if algorithm.endswith(baseline_CBV):
                baseline_name = algorithm
                break
        # pop out the baseline data
        baseline = datas.pop(baseline_name, None) if baseline_name is not None else None
        for i in range(max(num_algorithm-1, 1)):
            for row, (scenario_name, data) in enumerate(baseline.items()):
                # sns.kdeplot(data, color=color_list[0], ax=axs[row, i], alpha=0.8, label=baseline_name, fill=True, linewidth=1)
                axs[row, i].hist(data, density=density, bins=bins, alpha=0.75, label=baseline_name, color=(144/255, 190/255, 224/255))
                axs[row, i].set_title(scenario_name, fontsize=10)
                axs[row, i].set_xlabel(f'{data_name}')
                axs[row, i].set_ylabel('Frequency')
                axs[row, i].legend(fontsize=7, loc='best')

        # plot the rest algorithm
        for i, (algorithm, scenario) in enumerate(datas.items()):
            for row, (scenario_name, data) in enumerate(scenario.items()):
                # sns.kdeplot(data, color=color_list[i+1], ax=axs[row, i], alpha=0.7, label=algorithm, fill=True, linewidth=1)
                axs[row, i].hist(data, density=density, bins=bins, alpha=0.6, label=algorithm, color=color_list[i])
                axs[row, i].legend(fontsize=7, loc='best')

        plt.tight_layout()
        clean_string = re.sub(r'\(.*?\)', '', data_name)
        clean_string = clean_string.strip()
        data_save_name = clean_string.replace(' ', '_')
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
    TTC_all_data = {}
    ego_dis_data = {}
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
                TTC_all_data[algorithm_title] = {}
                ego_dis_data[algorithm_title] = {}
                feasibility[algorithm_title] = {}
                unfeasible_rate[algorithm_title] = {}
                if cbv != 'standard':
                    collision_ratio[algorithm_title] = {}
                    collision_vel[algorithm_title] = {}
                    collision_impulse[algorithm_title] = {}

                for scenario_map in scenario_map_files:
                    scenario_map_path = osp.join(algorithm_path, scenario_map)
                    if osp.isdir(scenario_map_path):
                        files = os.listdir(scenario_map_path)
                        for file in files:
                            if file == 'collision_traj_info.pkl' and 'collision_traj' in args.data:
                                file_path = os.path.join(scenario_map_path, file)
                                with open(file_path, 'rb') as pickle_file:
                                    all_collision_data = pickle.load(pickle_file)
                                collision_ratio[algorithm_title][scenario_map] = all_collision_data['collision_ratio']
                                collision_impulse[algorithm_title][scenario_map] = all_collision_data['collision_impulse']
                                collision_vel[algorithm_title][scenario_map] = all_collision_data['collision_vel']
                            if file == 'all_traj_info.pkl' and 'all_traj' in args.data:
                                file_path = os.path.join(scenario_map_path, file)
                                with open(file_path, 'rb') as pickle_file:
                                    all_data = pickle.load(pickle_file)
                                feasibility[algorithm_title][scenario_map] = all_data['feasibility_Vs']
                                unfeasible_rate[algorithm_title][scenario_map] = all_data['unfeasible_rate']
                            if file == 'miss_traj_info.pkl' and 'miss_traj' in args.data:
                                file_path = os.path.join(scenario_map_path, file)
                                with open(file_path, 'rb') as pickle_file:
                                    all_miss_data = pickle.load(pickle_file)
                                PET_all_data[algorithm_title][scenario_map] = all_miss_data['PET']
                                TTC_all_data[algorithm_title][scenario_map] = all_miss_data['TTC']
                                ego_dis_data[algorithm_title][scenario_map] = all_miss_data['ego_dis']

    if 'collision_traj' in args.data:
        # collision ratio info
        plot_metric(collision_ratio, 'collision ratio')
        # collision impulse info
        collision_impulse_bins = np.linspace(0, 20, 10)
        draw_data(collision_impulse, 'Collision impulse (kN·s)', ROOT_DIR, bins=collision_impulse_bins, baseline_CBV='ppo')
        # collision velocity info
        collision_vel_bins = np.linspace(0, 10, 20)
        draw_data(collision_vel, 'Collision velocity (m/s)', ROOT_DIR, bins=collision_vel_bins, baseline_CBV='ppo')

    if 'all_traj' in args.data:
        # unfeasible rate
        plot_metric(unfeasible_rate, 'unfeasible_rate')
        # feasibility distribution
        feasibility_bins = np.linspace(-3, 5, 20)
        draw_data(feasibility, 'Feasibility value', ROOT_DIR, bins=feasibility_bins)

    if 'miss_traj' in args.data:
        # PET distribution
        PET_bins = np.linspace(0, 3, 30)
        draw_data(PET_all_data, 'Post encroachment time (s)', ROOT_DIR, bins=PET_bins)

        # TTC distribution
        TTC_bins = np.linspace(0, 5, 20)
        draw_data(TTC_all_data, 'Time to collision (s)', ROOT_DIR, bins=TTC_bins)

        # ego distance distribution
        ego_dis_bins = np.linspace(0, 10, 30)
        draw_data(ego_dis_data, 'Distance to ego vehicle (m)', ROOT_DIR, bins=ego_dis_bins)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))))
    parser.add_argument('--data', '-d', nargs='*', type=str, default=['collision_traj', 'miss_traj', 'all_traj'])
    args = parser.parse_args()

    main(args)





