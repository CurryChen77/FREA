#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：plot_learning_curve.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/4/7
"""
import pandas as pd
import seaborn as sns
import os.path as osp
import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt


def extract_and_combine_events(path_to_event_files, label):
    all_events_data = []
    for file_path in path_to_event_files:
        events_data = extract_tb_data(file_path)
        # extract the corresponding label data
        if label in events_data:
            all_events_data.extend(events_data[label])

    # sort by steps and keep the latest data
    all_events_data.sort(key=lambda x: x[0])  # sort by steps
    combined_data = {}
    for step, value in all_events_data:
        combined_data[step] = value  # only keep the latest data for each step
    # transform back to list and sort by steps
    combined_data = sorted(combined_data.items(), key=lambda x: x[0])
    return combined_data


def extract_tb_data(path_to_events_file):
    ea = EventAccumulator(path_to_events_file)
    ea.Reload()
    # only care about the scalars
    scalar_tags = ea.Tags()['scalars']
    data = {}
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]
    return data


def smooth(data, sm=1):
    ''' 平滑数据 '''
    if sm > 1:
        z = np.ones_like(data)
        y = np.ones(sm) * 1.0
        smoothed = np.convolve(y, data, "same") / np.convolve(y, z, "same")
        return smoothed
    else:
        return data


def read_df(df_path):
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        print('>> ' + '-' * 20, 'Reading learning curve data', '-' * 20)

    else:
        dataframes = []
        root_dir = osp.join(args.ROOT_DIR, 'log/train_scenario')
        # walk through all the dir in the root directory
        for root, dirs, files in os.walk(root_dir):
            event_files = [f for f in files if f.startswith('events.out.tfevents')]
            if event_files:
                path_elements = root.split(os.sep)
                algorithm, seed = path_elements[-2].split('_seed')
                scene = path_elements[-1]
                seed = int(seed)
                path_to_event_files = [os.path.join(root, f) for f in event_files]
                combined_events_data = extract_and_combine_events(path_to_event_files, label='Scenario_average_reward_per_step')
                for step, value in combined_events_data:
                    dataframes.append(pd.DataFrame({
                        'step': [step],
                        'value': [value],
                        'seed': [seed],
                        'algorithm': [algorithm],
                        'scene': [scene]
                    }))
        # merge all the dataframes
        df = pd.concat(dataframes, ignore_index=True)
        df.to_csv(osp.join(args.ROOT_DIR, 'eval_analysis/processed_data/learning_curve.csv'), index=False)
        print('>> ' + '-' * 20, 'Saving learning curve data', '-' * 20)

    return df


def main(args):
    # read the learning curve data
    df_path = osp.join(args.ROOT_DIR, 'eval_analysis/processed_data/learning_curve.csv')
    df = read_df(df_path)

    sns.set(style="darkgrid")

    # create subplot
    scenes = df['scene'].drop_duplicates()
    num_plots = len(scenes)
    subplots_height = 5
    aspect_ratio = 1.2
    figsize = (subplots_height * aspect_ratio * num_plots, subplots_height)
    fig, axs = plt.subplots(1, num_plots, figsize=figsize, squeeze=False)  # make sure axs are always 2D

    for index, scene in enumerate(scenes):
        ax = axs[0, index]
        scene_df = df[df['scene'] == scene]

        # handel algorithm in each scene
        for algorithm, alg_df in scene_df.groupby('algorithm'):
            smoothed_dfs = []
            for seed, seed_df in alg_df.groupby('seed'):
                seed_df = seed_df.sort_values('step')
                seed_df['smoothed_value'] = smooth(seed_df['value'].values, sm=200)
                smoothed_dfs.append(seed_df)

            # merge the smoothed data
            smoothed_df = pd.concat(smoothed_dfs)
            # plot the mean value and the trust region
            sns.lineplot(ax=ax, data=smoothed_df, x='step', y='smoothed_value',
                         estimator='mean', errorbar=('ci', 95), label=algorithm,
                         err_kws={"alpha": 0.2, "linewidth": 0.1})  # error_kws: the parameter for the trust region

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[0:], labels=labels[0:], title="Algorithms", loc="best", fontsize=8, title_fontsize=10)

        ax.set_title(f'{scene}')
        ax.set_xlabel('Step')
        ax.set_ylabel('Step Average Reward')

    plt.tight_layout()
    save_dir = osp.join(args.ROOT_DIR, f'eval_analysis/figures/Learning_curve.png')
    plt.savefig(save_dir, dpi=600)
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))))
    args = parser.parse_args()

    main(args)