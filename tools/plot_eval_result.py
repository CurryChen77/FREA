#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：plot_eval_result.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/2/28
"""

import pickle
import seaborn as sns
import os.path as osp
import os
import pandas as pd


import matplotlib
from matplotlib import pyplot as plt
from tabulate import tabulate


def read_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def draw_hist(velocity, acc, ego_dis, title):
    assert len(velocity) == len(acc) == len(ego_dis) == len(title), 'the input len should be the same'
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    num_algorithm = len(title)
    num_map = len(velocity[0])
    color_list = ['darkblue', 'red']

    fig, axs = plt.subplots(num_map, 3, figsize=(9, 9))

    for i in range(num_algorithm):
        for row, map_name in enumerate(velocity[i].keys()):

            sns.kdeplot(velocity[i][map_name], color=color_list[i], ax=axs[row, 0], label=title[i], alpha=0.3, fill=True, linewidth=0.5)
            axs[row, 0].set_title(map_name, fontsize=10)
            axs[row, 0].set_xlabel('Velocity')
            axs[row, 0].set_ylabel('Density')

            sns.kdeplot(acc[i][map_name], color=color_list[i], ax=axs[row, 1], label=title[i], alpha=0.3, fill=True, linewidth=0.5)
            axs[row, 1].set_title(map_name, fontsize=10)
            axs[row, 1].set_xlabel('Acc')
            axs[row, 1].set_ylabel('Density')

            sns.kdeplot(ego_dis[i][map_name], color=color_list[i], ax=axs[row, 2], label=title[i], alpha=0.3, fill=True, linewidth=0.5)
            axs[row, 2].set_title(map_name, fontsize=10)
            axs[row, 2].set_xlabel('Ego distance')
            axs[row, 2].set_ylabel('Density')

    for i in range(num_map):
        for j in range(3):
            axs[i, j].legend(fontsize=8, loc='upper right')

    plt.tight_layout()

    plt.show()


def main(ROOT_DIR, args):
    algorithm_files = os.listdir(ROOT_DIR)
    all_velocity = []
    all_acc = []
    all_ego_dis = []
    algorithm_titles = []
    all_results = {}
    for algorithm in algorithm_files:
        ego, cbv, select, seed = algorithm.split('_')
        algorithm_path = osp.join(ROOT_DIR, algorithm)
        if osp.isdir(algorithm_path):
            scenario_map_files = os.listdir(algorithm_path)
            velocity = {name: [] for name in scenario_map_files}
            acc = {name: [] for name in scenario_map_files}
            ego_dis = {name: [] for name in scenario_map_files}
            algorithm_title = f"Ego:{ego} CBV:{cbv}"
            algorithm_titles.append(algorithm_title)

            for scenario_map in scenario_map_files:
                scenario_map_path = osp.join(algorithm_path, scenario_map)
                scenario_map_title = f'\n{scenario_map}'
                if osp.isdir(scenario_map_path):
                    files = os.listdir(scenario_map_path)
                    if args.metric and 'results.pkl' in files:
                        results = read_pickle_file(osp.join(scenario_map_path, 'results.pkl'))
                        all_results[algorithm_title + scenario_map_title] = results
                        # print('>> ' + '-' * 70)
                        # print(algorithm_title + scenario_map_title)
                        # print('>> ' + '-' * 70)
                        # print(tabulate(results.items(), headers=['Metric', 'Value'], tablefmt='grid'))
                    if 'records.pkl' in files:
                        record = read_pickle_file(osp.join(scenario_map_path, 'records.pkl'))
                        for scenario_data in record.values():
                            for step in scenario_data:
                                if cbv == 'standard' and 'BVs_velocity' in step.keys():
                                    velocity[scenario_map].append(step['BVs_velocity'])
                                    acc[scenario_map].append(step['BVs_acc'])
                                    ego_dis[scenario_map].append(step['BVs_ego_dis'])
                                elif 'CBVs_velocity' in step.keys():
                                    velocity[scenario_map].extend(step['CBVs_velocity'].values())
                                    acc[scenario_map].extend(step['CBVs_acc'].values())
                                    ego_dis[scenario_map].extend(step['CBVs_ego_dis'].values())
            all_velocity.append(velocity)
            all_acc.append(acc)
            all_ego_dis.append(ego_dis)
    if args.metric:
        df = pd.DataFrame(all_results)
        sns.heatmap(df, annot=True, cmap='coolwarm', fmt='.1f', linewidths=.5)
        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.yticks(rotation=45, va='center', fontsize=7)
        plt.title('Performance Metrics')
        plt.tight_layout()
        plt.show()
    draw_hist(all_velocity, all_acc, all_ego_dis, algorithm_titles)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', '-m', action='store_true')
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__)))))
    args = parser.parse_args()

    ROOT_DIR = osp.join(args.ROOT_DIR, 'log/eval')

    main(ROOT_DIR, args)




