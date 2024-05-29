#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：plot_learning_curve.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/4/7
"""
import re

import pandas as pd
import seaborn as sns
import os.path as osp
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


def smooth(data, sm=1):
    """
        smooth the input data
    """
    if sm > 1:
        z = np.ones_like(data)
        y = np.ones(sm) * 1.0
        smoothed = np.convolve(y, data, "same") / np.convolve(y, z, "same")
        return smoothed
    else:
        return data


def read_df(df_path, ROOT_DIR):
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        print('>> ' + '-' * 20, 'Reading learning curve data', '-' * 20)

    else:
        dataframes = []
        pattern = r"run-(.+?)_seed(\d+)_Scenario(\d+)_Town(\d+)-tag-(.+)\.csv"
        root_dir = osp.join(ROOT_DIR, 'eval_analysis/train_result/')
        # walk through all the dir in the root directory
        for root, dirs, files in os.walk(root_dir):
            file_names = [f for f in files if f.startswith('run-')]
            for file_name in file_names:
                match = re.search(pattern, file_name)
                if match:
                    algorithm = match.group(1)
                    seed = int(match.group(2))
                    scenario = f"Scenario{match.group(3)}_Town{match.group(4)}"
                    label = match.group(5)
                    file_path = os.path.join(root, file_name)
                    data = pd.read_csv(file_path)

                    # data = data[data['Step'] % 2 == 0]

                    dataframes.append(pd.DataFrame({
                        'step': data['Step'],
                        'value': data['Value'],
                        'seed': np.repeat(seed, len(data)),
                        'algorithm': np.repeat(algorithm, len(data)),
                        'scene': np.repeat(scenario, len(data)),
                        'label': np.repeat(label, len(data))
                }))
                else:
                    print("no matching")

        # merge all the dataframes
        df = pd.concat(dataframes, ignore_index=True)
        df.to_csv(osp.join(args.ROOT_DIR, 'eval_analysis/train_result/learning_curve.csv'), index=False)
        print('>> ' + '-' * 20, 'Saving learning curve data', '-' * 20)

    return df


def main(args):
    # read the learning curve data
    df_path = osp.join(args.ROOT_DIR, 'eval_analysis/train_result/learning_curve.csv')
    all_df = read_df(df_path, args.ROOT_DIR)
    font_props = font_manager.FontProperties(family='Times New Roman', size=12)

    sns.set(style="darkgrid")
    labels = all_df['label'].drop_duplicates()

    for label in labels:
        # Filter dataframe for current label
        df = all_df[all_df['label'] == label]

        # create subplot
        scenes = df['scene'].drop_duplicates()
        num_plots = len(scenes)
        num_algo = len(df['algorithm'].drop_duplicates())
        subplots_height = 5
        aspect_ratio = 1.2
        figsize = (subplots_height * aspect_ratio * num_plots, subplots_height)
        fig, axs = plt.subplots(1, num_plots, figsize=figsize, squeeze=False)  # make sure axs are always 2D
        all_labels = []
        all_handles = []

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
                split_name = algorithm.split('_')
                ego = split_name[0]
                cbv = '-'.join(split_name[1:-1])
                algo_label = f"AV:{ego} CBV:{cbv}"
                # plot the mean value and the trust region
                sns.lineplot(ax=ax, data=smoothed_df, x='step', y='smoothed_value',
                            estimator='mean', errorbar=('ci', 95),
                            err_kws={"alpha": 0.2, "linewidth": 0.1})  # error_kws: the parameter for the trust region
                all_labels.append(algo_label)
                all_handles.append(ax.lines[-1])

            # ax.legend(handles=handles[0:], labels=labels[0:], title="Algorithm",
            #           loc="best", prop=font_props, title_fontproperties=title_font_props)
            ax.set_title(f'{scene}', fontfamily='Times New Roman', fontsize=16)
            ax.set_xlabel('Train step', fontfamily='Times New Roman', fontsize=16)
            ax.set_ylabel('Episode Return', fontfamily='Times New Roman', fontsize=16)

        unique_handles_labels = dict(zip(all_labels, all_handles))
        unique_labels, unique_handles = zip(*unique_handles_labels.items())
        fig.legend(handles=unique_handles, labels=unique_labels, loc='lower center', ncol=num_algo, prop=font_props, fontsize=16)

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)

        save_dir = osp.join(args.ROOT_DIR, f'eval_analysis/figures/Episode_return.png')
        plt.savefig(save_dir, dpi=600)
        plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))))
    args = parser.parse_args()

    main(args)