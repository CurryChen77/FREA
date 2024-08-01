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
from collections import OrderedDict
import re
import matplotlib
from matplotlib import pyplot as plt


def classified_data_by_ego(data):
    classified_by_ego = defaultdict(dict)

    for key, value in data.items():
        match = re.search(r"AV:(\w+)", key)
        if match:
            ego = match.group(1)  # extract ego
            classified_by_ego[ego][key] = value

    classified_by_ego = dict(classified_by_ego)
    return classified_by_ego


def draw_data(All_data, data_name, ROOT_DIR, bins, baseline_CBV='Standard', density=True):
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    All_data_per_ego = classified_data_by_ego(All_data)
    y_label = 'Probability Density' if density else 'Frequency'
    for ego_type, datas in All_data_per_ego.items():
        num_algorithm = len(datas.keys())
        num_scenarios = max(len(scenarios) for scenarios in datas.values())
        color_list = sns.color_palette("flare", n_colors=num_algorithm-1)
        baseline_name = None
        subplots_height = 5
        aspect_ratio = 0.85
        num_cols = max(num_algorithm-1, 1)
        figsize = (subplots_height * aspect_ratio * num_cols, subplots_height)
        all_handles = []
        all_labels = []

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
                handles, labels = axs[row, i].get_legend_handles_labels()
                all_handles.extend(handles)
                all_labels.extend(labels)

        # rearrange the data order
        desired_order = ['Standard', 'PPO', 'FPPO-RS', 'FREA']
        ordered_data = OrderedDict()
        for order_key in desired_order:
            for key in datas:
                if key.endswith(order_key):
                    ordered_data[key] = datas[key]
                    break

        # plot the rest algorithm
        for i, (algorithm, scenario) in enumerate(ordered_data.items()):
            for row, (scenario_name, data) in enumerate(scenario.items()):
                # sns.kdeplot(data, color=color_list[i+1], ax=axs[row, i], alpha=0.7, label=algorithm, fill=True, linewidth=1)
                axs[row, i].hist(data, density=density, bins=bins, alpha=0.6, label=algorithm, color=color_list[i])
                axs[row, i].set_title(scenario_name, fontsize=14)
                axs[row, i].set_xlabel(f'{data_name}', fontsize=14)
                axs[row, i].set_ylabel(y_label, fontsize=13)
                handles, labels = axs[row, i].get_legend_handles_labels()
                all_handles.extend(handles)
                all_labels.extend(labels)

        unique_handles_labels = dict(zip(all_labels, all_handles))
        unique_labels, unique_handles = zip(*unique_handles_labels.items())
        fig.legend(handles=unique_handles, labels=unique_labels, loc='lower center', ncol=num_algorithm, fontsize=14)

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)

        clean_string = re.sub(r'\(.*?\)', '', data_name)
        clean_string = clean_string.strip()
        data_save_name = clean_string.replace(' ', '_')
        save_dir = osp.join(ROOT_DIR, f'eval_analysis/figures/{ego_type}-{data_save_name}.png')
        plt.savefig(save_dir, dpi=600)
        plt.show()


def plot_metric(result, name, activate_per=True):
    print('>> ' + '-' * 10, str(name), '-' * 10)
    for algorithm, scenario_data in result.items():
        for scenario_name, data in scenario_data.items():
            if activate_per:
                percentage = round(data * 100, 2)
            else:
                percentage = round(data, 2)
            print(f">> {algorithm} in {scenario_name}: {percentage}")
    print('>> ' + '-' * 10, str(name), '-' * 10)


def process_record(args):
    ROOT_DIR = args.ROOT_DIR
    # only plot the CBV evaluation pretrained with expert ego
    base_dir = osp.join(ROOT_DIR, 'eval_analysis/processed_data/eval_cbv_pretrained_with_expert')
    algorithm_files = os.listdir(base_dir)
    PET_all_data = {}
    TTC_all_data = {}
    ego_dis_data = {}
    infeasible_ratio = {}
    feasibility_Vs = {}
    infeasible_distance = {}
    all_infeasible_distance = {}
    collision_ratio = {}
    collision_vel = {}
    collision_impulse = {}

    for algorithm in algorithm_files:
        if osp.isdir(osp.join(base_dir, algorithm)):
            split_name = algorithm.split('_')
            ego = split_name.pop(0)
            seed = split_name.pop(-1)
            select = split_name.pop(-1)
            cbv = '-'.join(split_name) if len(split_name) > 1 else split_name[0]
            algorithm_path = osp.join(base_dir, algorithm)
            if osp.isdir(algorithm_path):
                scenario_map_files = os.listdir(algorithm_path)
                # the specific ego and CBV method
                if str(ego) == 'expert':
                    Ego_name = 'Expert'
                elif str(ego) == 'plant':
                    Ego_name = 'PlanT'
                else:
                    Ego_name = str(ego)
                if str(cbv) == 'fppo-adv':
                    CBV_name = 'FREA'
                elif str(cbv) == 'fppo-rs':
                    CBV_name = 'FPPO-RS'
                elif str(cbv) == 'ppo':
                    CBV_name = 'PPO'
                else:
                    CBV_name = 'Standard'
                algorithm_title = f"AV:{Ego_name}  CBV:{CBV_name}"

                PET_all_data[algorithm_title] = {}
                TTC_all_data[algorithm_title] = {}
                ego_dis_data[algorithm_title] = {}
                if cbv != 'standard':
                    collision_ratio[algorithm_title] = {}
                    collision_vel[algorithm_title] = {}
                    collision_impulse[algorithm_title] = {}
                    infeasible_ratio[algorithm_title] = {}
                    feasibility_Vs[algorithm_title] = {}
                    infeasible_distance[algorithm_title] = {}
                    all_infeasible_distance[algorithm_title] = {}

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
                                PET_all_data[algorithm_title][scenario_map] = all_data['PET']
                                TTC_all_data[algorithm_title][scenario_map] = all_data['TTC']
                                ego_dis_data[algorithm_title][scenario_map] = all_data['ego_dis']
                                if cbv != 'standard':
                                    infeasible_ratio[algorithm_title][scenario_map] = all_data['infeasible_ratio']
                                    feasibility_Vs[algorithm_title][scenario_map] = all_data['feasibility_Vs']
                                    infeasible_distance[algorithm_title][scenario_map] = all_data['infeasible_distance']
                                    all_infeasible_distance[algorithm_title][scenario_map] = all_data['all_infeasible_distance']

    if 'collision_traj' in args.data:
        # collision ratio info
        plot_metric(collision_ratio, 'Collision Ratio (%)')
        # collision impulse info
        collision_impulse_bins = np.linspace(0, 20, 10)
        draw_data(collision_impulse, 'Collision Impulse (kN·s)', ROOT_DIR, bins=collision_impulse_bins, baseline_CBV='PPO')
        # collision velocity info
        collision_vel_bins = np.linspace(0, 10, 20)
        draw_data(collision_vel, 'Collision Velocity (m/s)', ROOT_DIR, bins=collision_vel_bins, baseline_CBV='PPO')

    if 'all_traj' in args.data:
        # PET distribution
        PET_bins = np.linspace(0, 3, 30)
        draw_data(PET_all_data, 'Post Encroachment Time (s)', ROOT_DIR, bins=PET_bins, baseline_CBV='Standard')

        # TTC distribution
        TTC_bins = np.linspace(0, 5, 20)
        draw_data(TTC_all_data, 'Time to Collision (s)', ROOT_DIR, bins=TTC_bins, baseline_CBV='Standard')

        # ego distance distribution
        ego_dis_bins = np.linspace(0, 10, 30)
        draw_data(ego_dis_data, 'Distance to Ego Vehicle (m)', ROOT_DIR, bins=ego_dis_bins, baseline_CBV='Standard')

        plot_metric(infeasible_ratio, 'Infeasible Ratio (%)')
        fea_bins = np.linspace(0, 10, 20)
        draw_data(feasibility_Vs, 'Feasibility Values', ROOT_DIR, bins=fea_bins, baseline_CBV='PPO')

        # infeasible distance
        plot_metric(infeasible_distance, 'Infeasible Distance (m)', activate_per=False)
        infeasible_distance_bins = np.linspace(0, 20, 10)
        draw_data(all_infeasible_distance, 'Infeasible Distance Distribution (m)', ROOT_DIR, bins=infeasible_distance_bins, baseline_CBV='PPO')


def list_directories(path):
    items = os.listdir(path)
    directories = [item for item in items if os.path.isdir(os.path.join(path, item))]
    return directories


def process_result(args):
    ROOT_DIR = args.ROOT_DIR
    eval_dirs = list_directories(osp.join(ROOT_DIR, 'log/eval'))
    for eval_dir in eval_dirs:
        # plot all the result of in the evaluation dir
        print(f'>> --------------------Processing {eval_dir}--------------------')
        base_dir = os.path.join(osp.join(ROOT_DIR, 'log/eval', eval_dir))

        algorithm_files = os.listdir(base_dir)
        for algorithm in algorithm_files:
            if osp.isdir(osp.join(base_dir, algorithm)):
                split_name = algorithm.split('_')
                ego = split_name.pop(0)
                seed = split_name.pop(-1)
                select = split_name.pop(-1)
                cbv = '-'.join(split_name) if len(split_name) > 1 else split_name[0]
                algorithm_path = osp.join(base_dir, algorithm)
                if osp.isdir(algorithm_path):
                    scenario_map_files = os.listdir(algorithm_path)
                    # the specific ego and CBV method
                    if str(ego) == 'expert':
                        Ego_name = 'Expert'
                    elif str(ego) == 'plant':
                        Ego_name = 'PlanT'
                    else:
                        Ego_name = str(ego)
                    if str(cbv) == 'fppo-adv':
                        CBV_name = 'FREA'
                    elif str(cbv) == 'fppo-rs':
                        CBV_name = 'FPPO-RS'
                    elif str(cbv) == 'ppo':
                        CBV_name = 'PPO'
                    else:
                        CBV_name = 'Standard'
                    algorithm_title = "AV:{:<8} CBV:{:<10}".format(str(Ego_name), CBV_name)
                    for scenario_map in scenario_map_files:
                        scenario_map_path = osp.join(algorithm_path, scenario_map)
                        if osp.isdir(scenario_map_path):
                            files = os.listdir(scenario_map_path)
                            for file in files:
                                if file == 'results.pkl':
                                    file_path = os.path.join(scenario_map_path, file)
                                    with open(file_path, 'rb') as pickle_file:
                                        all_results = pickle.load(pickle_file)
                                        plot_results(algorithm_title, all_results, scenario_map)
        print(f'>> --------------------Finishing {eval_dir}--------------------')


def plot_results(name, results, scenario_map, activate_per=True):
    print('>>', "{:<8}: {:<17}".format(str(name), str(scenario_map)), end=" ")
    for key, value in results.items():
        if activate_per:
            percentage = round(value * 100, 2)
        else:
            percentage = round(value, 2)
        print("{:<10}: {:<4}".format(key, percentage), end=" ")
    print('')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))))
    parser.add_argument('--data', '-d', nargs='*', type=str, default=['collision_traj', 'all_traj'])
    parser.add_argument('--mode', '-m', type=str, default=['record', 'result'], nargs='+', choices=['record', 'result'])

    args = parser.parse_args()
    # the analysis of the record
    if 'record' in args.mode:
        process_record(args)

    # the evaluation results
    if 'result' in args.mode:
        process_result(args)





