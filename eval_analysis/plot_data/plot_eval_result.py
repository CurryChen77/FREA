#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：plot_eval_result.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/2/28
"""

import pickle
import numpy as np
import seaborn as sns
import os.path as osp
import os
import pandas as pd
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


def draw_PET(PET_data, args):

    matplotlib.rcParams['font.family'] = 'Times New Roman'
    PET_data_per_ego = classified_data_by_ego(PET_data)
    for ego_type, data in PET_data_per_ego.items():
        num_algorithm = len(data.keys())
        num_scenarios = max(len(scenarios) for scenarios in data.values())
        color_list = sns.color_palette("coolwarm", n_colors=num_algorithm)
        baseline_name = None
        fig, axs = plt.subplots(nrows=num_scenarios, ncols=max(num_algorithm-1, 1), squeeze=False)
        # plot the baseline
        for algorithm, scenario in data.items():
            baseline_name = algorithm if algorithm.endswith('standard') else None
        # pop out the baseline data
        baseline = data.pop(baseline_name, None) if baseline_name is not None else None
        bins = np.linspace(0, 5, 50)
        for i in range(max(num_algorithm-1, 1)):
            for row, (scenario_name, PET) in enumerate(baseline.items()):
                # sns.kdeplot(PET, color=color_list[0], ax=axs[row, i], alpha=0.8, label=baseline_name, fill=True, linewidth=1)
                axs[row, i].hist(PET, density=True, bins=bins, alpha=0.75, label=baseline_name, color=color_list[0])
                axs[row, i].set_title(scenario_name, fontsize=10)
                axs[row, i].set_xlabel('Post encroachment time')
                axs[row, i].set_ylabel('Frequency')
                axs[row, i].legend(fontsize=8, loc='upper right')

        # plot the rest algorithm
        for i, (algorithm, scenario) in enumerate(data.items()):
            for row, (scenario_name, PET) in enumerate(scenario.items()):
                # sns.kdeplot(PET, color=color_list[i+1], ax=axs[row, i], alpha=0.7, label=algorithm, fill=True, linewidth=1)
                axs[row, i].hist(PET, density=True, bins=bins, alpha=0.75, label=algorithm, color=color_list[i+1])
                axs[row, i].legend(fontsize=8, loc='upper right')

        plt.tight_layout()
        save_dir = osp.join(args.ROOT_DIR, 'eval_analysis/figures/PET.png')
        plt.savefig(save_dir, dpi=600)
        plt.show()


def draw_metric(all_results):
    df = pd.DataFrame(all_results) * 100
    sns.heatmap(df, annot=True, cmap='coolwarm', fmt='.0f', linewidths=.5)
    plt.xticks(rotation=45, ha='center', fontsize=7)
    plt.yticks(va='center', fontsize=7)
    plt.title('Performance Metrics')
    plt.tight_layout()
    plt.show()


def main(args):
    base_dir = osp.join(args.ROOT_DIR, 'eval_analysis/processed_data')
    algorithm_files = os.listdir(base_dir)
    PET_data = {}

    for algorithm in algorithm_files:
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
            PET_data[algorithm_title] = {}

            for scenario_map in scenario_map_files:
                scenario_map_path = osp.join(algorithm_path, scenario_map)
                if osp.isdir(scenario_map_path):
                    files = os.listdir(scenario_map_path)
                    for file in files:
                        # processing PET data
                        if file == 'PET.npy':
                            file_path = os.path.join(scenario_map_path, file)
                            PET_data[algorithm_title][scenario_map] = np.load(file_path)

    # draw PET data
    draw_PET(PET_data, args)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', '-m', action='store_true')
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))))
    args = parser.parse_args()

    main(args)




