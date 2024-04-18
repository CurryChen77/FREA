#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：plot_feasibility.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/1/25
"""
import os
import re

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from matplotlib import colors
import os.path as osp
import torch
from matplotlib.colors import LogNorm, ListedColormap

from safebench.feasibility import FEASIBILITY_LIST
from safebench.feasibility.dataset import OffRLDataset
from safebench.gym_carla.envs.utils import linear_map
from safebench.util.logger import Logger
from safebench.util.run_util import load_config
from safebench.util.torch_util import set_torch_variable, set_seed, CUDA, CPU
from eval_analysis.plot_data.plot_learning_curve import smooth, extract_and_combine_events


def calculate_collision_rate(done, collision):
    '''
        the collision is counted considering the episode collision
        (if collision happens in this episode, then the collision is counted as collision_episode)
    '''
    total_episodes = 0
    collision_episodes = 0
    has_collision = False
    for d, c in zip(done, collision):
        if d:  # episode end
            total_episodes += 1
            if has_collision or c:
                collision_episodes += 1
            has_collision = False
        elif c:
            has_collision = True
    # count the total collision
    collision_rate = collision_episodes / total_episodes if total_episodes != 0 else None
    return collision_rate


def plot_feasibility_data_distribution(args):
    # the route of the data needs to be processed
    data_file_path = osp.join(args.ROOT_DIR, args.data_route, args.scenario_map, args.data_filename)
    dataset = OffRLDataset(data_file_path, device='CPU')

    obs = dataset.dataset_dict['obs']
    action = dataset.dataset_dict['actions']
    ego_min_dis = dataset.dataset_dict['ego_min_dis']
    ego_collision_percentage = calculate_collision_rate(dataset.dataset_dict['dones'], dataset.dataset_dict['ego_collide']) * 100

    # x, y position of the closest point
    x_coords = obs[:, 1, 0].flatten()
    y_coords = obs[:, 1, 1].flatten()
    both_zero_positions = np.logical_and(abs(x_coords) < 0.001, abs(y_coords) < 0.001)
    non_zero_x = x_coords[~both_zero_positions]
    non_zero_y = y_coords[~both_zero_positions]

    # relative yaw, speed of the closest point
    yaw = obs[:, 1, 4].flatten()
    speed = obs[:, 1, 5].flatten()
    both_zero_positions = abs(yaw) < 0.001
    non_zero_yaw = yaw[~both_zero_positions]
    non_zero_speed = speed[~both_zero_positions]

    # throttle and speed
    throttle = action[:, 0]
    steering_angle = action[:, 1]

    matplotlib.rcParams['font.family'] = 'Times New Roman'

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    color_palette = sns.color_palette("Blues")
    cmap = ListedColormap(color_palette)

    _, _, _, x_y_img = axs[0, 0].hist2d(non_zero_x, non_zero_y, bins=60, cmap=cmap, norm=LogNorm(), alpha=0.9)
    axs[0, 0].set_title('Closest Vehicle Position (x, y)', fontsize=12)
    axs[0, 0].set_xlabel('X Coordinate')
    axs[0, 0].set_ylabel('Y Coordinate')
    fig.colorbar(x_y_img, ax=axs[0, 0])

    axs[0, 1].hist(ego_min_dis, density=True, bins=30, alpha=0.7, color=color_palette[len(color_palette) // 2])
    # sns.kdeplot(ego_min_dis, color=color_palette[len(color_palette) // 2], ax=axs[0, 1], alpha=0.6, fill=True, linewidth=1.2)
    axs[0, 1].set_title('Closest dis between Ego and BVs', fontsize=12)
    axs[0, 1].set_xlabel('Closest distance')
    axs[0, 1].set_ylabel('Frequency')
    text = 'Ego episode collision rate: {:.2f}%'.format(ego_collision_percentage)
    axs[0, 1].legend(labels=[text], loc='upper right', fontsize=12, labelcolor='red')

    _, _, _, yaw_speed_img = axs[1, 0].hist2d(non_zero_yaw, non_zero_speed, bins=60, cmap=cmap, norm=LogNorm(), alpha=0.9)
    axs[1, 0].set_title('Closest Vehicle yaw and Speed', fontsize=12)
    axs[1, 0].set_xlabel('Relative yaw')
    axs[1, 0].set_ylabel('Speed')
    fig.colorbar(yaw_speed_img, ax=axs[1, 0])

    _, _, _, throttle_steering_angle_img = axs[1, 1].hist2d(throttle, steering_angle, bins=45, cmap=cmap, norm=LogNorm(), alpha=0.9)
    axs[1, 1].set_title('Ego Vehicle Throttle and Speed', fontsize=12)
    axs[1, 1].set_xlabel('Throttle')
    axs[1, 1].set_ylabel('Steering angle')
    axs[1, 1].set_ylim([-0.75, 0.75])
    fig.colorbar(throttle_steering_angle_img, ax=axs[1, 1])

    plt.tight_layout()
    save_dir = osp.join(args.ROOT_DIR, 'safebench/feasibility/figures/feasibility_data_distribution.png')
    plt.savefig(save_dir, dpi=600)
    plt.show()


def rotated_rectangles_overlap(rect1, rect2):
    def get_corners(x, y, width, height, angle):
        # Get the four corners of the rectangle
        corners = np.array([[-width / 2, -height / 2],
                            [width / 2, -height / 2],
                            [width / 2, height / 2],
                            [-width / 2, height / 2]])

        # Rotate the corners
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        rotated_corners = np.dot(corners, rotation_matrix.T)

        # Translate the corners to the specified position
        translated_corners = rotated_corners + np.array([x, y])

        return translated_corners

    corners1 = get_corners(*rect1)
    corners2 = get_corners(*rect2)

    # Check for overlap using the Separating Axis Theorem (SAT)
    for axis in range(2):
        # Project the corners onto the axis
        projections1 = corners1[:, axis]
        projections2 = corners2[:, axis]

        # Check for overlap on the axis
        overlap = (np.max(projections1) >= np.min(projections2) and np.max(projections2) >= np.min(projections1))

        if not overlap:
            return False  # No overlap on this axis, rectangles do not overlap

    return True  # Overlapping on both axes, rectangles overlap


def generate_random_rectangle(x_range, y_range, width=4.9, height=2.12, angle=0.0):
    x = np.random.uniform(x_range[0] + width/2, x_range[1] - width/2)  # Adjust the range based on your plot size
    y = np.random.uniform(y_range[0] + height/2, y_range[1] - height/2)  # Adjust the range based on your plot size
    return x, y, width, height, angle


def generate_non_overlapping_rectangles(num_rectangles, x_range, y_range, width=4.9, height=2.12, yaw_angle=(-3.14, 3.14)):
    rectangles = [[0, 0, width, height, 0]]
    index = 0
    while len(rectangles) < num_rectangles:
        rectangle = generate_random_rectangle(x_range, y_range, width, height, yaw_angle[index])
        if all(not rotated_rectangles_overlap(rectangle, existing) for existing in rectangles):
            rectangles.append(rectangle)
            index += 1
    return rectangles


def generate_ego_obs(ego_speed, yaw_angle=(-0.0, 0.0), actor_num=3, x_range=(-23, 23), y_range=(-9, 9), width=4.9, height=2.12, speed_range=(1, 6)):
    assert len(yaw_angle) == actor_num - 1  # the yaw angle for the BVs should == actor number -1
    BV_list = np.zeros((actor_num, 6), dtype=np.float32)
    rectangles = generate_non_overlapping_rectangles(actor_num, x_range, y_range, width, height, yaw_angle)
    BV_list[:, 0:5] = rectangles
    BV_list[0:, 5] = linear_map(np.random.rand(actor_num), original_range=(0, 1), desired_range=speed_range)
    BV_list[0, 5] = ego_speed
    return BV_list


def rearrange_the_actor_order(batch_ego_obs):
    for i in range(batch_ego_obs.shape[0]):
        # if the third row's distance is closer than the second row's distance, then switch row 2 and row 3
        if np.linalg.norm(batch_ego_obs[i, 2, :]) < np.linalg.norm(batch_ego_obs[i, 1, :]):
            batch_ego_obs[i, [1, 2]] = batch_ego_obs[i, [2, 1]]
    return batch_ego_obs


def plot_feasibility_region(ax, agent, ego_obs, ego_speed, spatial_interval=10, actor_num=3, x_range=(-25, 25), y_range=(-10, 10)):
    from matplotlib.patches import Rectangle, Arrow

    # the fake ego x, y coordinates
    x = np.linspace(x_range[0], x_range[1], spatial_interval)
    y = np.linspace(y_range[0], y_range[1], spatial_interval)
    x_grid, y_grid = np.meshgrid(x, y)
    flatten_x = x_grid.ravel()
    flatten_y = y_grid.ravel()
    batch_ego_obs = np.tile(ego_obs.reshape(1, actor_num, 6), (spatial_interval**2, 1, 1))
    # transform the fake ego x, y coordinates into the relative coordinates of the surrounding vehicles
    ego_x = np.tile(flatten_x[:, np.newaxis], (1, actor_num-1))
    ego_y = np.tile(flatten_y[:, np.newaxis], (1, actor_num-1))
    batch_ego_obs[:, 1:, 0] -= ego_x
    batch_ego_obs[:, 1:, 1] -= ego_y
    batch_ego_obs = rearrange_the_actor_order(batch_ego_obs)
    batch_ego_obs[:, 0, 5] = np.ones_like(flatten_x) * ego_speed

    # get the corresponding feasibility values
    feasibility_value = CPU(agent.get_feasibility_Vs(CUDA(torch.FloatTensor(batch_ego_obs))))
    array_value = np.asarray(feasibility_value).reshape(x_grid.shape)

    norm = colors.Normalize(vmin=-8, vmax=4)

    ct = ax.contourf(
        x_grid, y_grid, array_value,
        norm=norm,
        levels=15,
        cmap='rainbow',
        alpha=0.5
    )
    # draw the '0' contour line
    if np.any(array_value <= 0):
        ct_line = ax.contour(
            x_grid, y_grid, array_value,
            levels=[0], colors='#32ABD6',
            linewidths=2.0, linestyles='solid'
        )
        ax.clabel(ct_line, inline=True, fontsize=10, fmt=r'0')

    cb = plt.colorbar(ct, ax=ax, shrink=0.8, pad=0.01)
    cb.ax.tick_params(labelsize=7)

    # plot all the vehicles
    for i, Vehicle in enumerate(ego_obs):
        # the x,y in Rectangle in the bottom left corner
        color = 'r' if i == 0 else (0.30, 0.52, 0.74)
        # the angle of the Rectangle is under right-handed coordinate system, but the yaw angle in carla is left-handed coordinate system
        rectangle = Rectangle((Vehicle[0]-Vehicle[2] / 2, Vehicle[1]-Vehicle[3] / 2), width=Vehicle[2], height=Vehicle[3], angle=-np.degrees(Vehicle[4]), rotation_point='center', fill=True, alpha=0.5, color=color)
        angle_rad = Vehicle[4]
        speed = ego_speed if i == 0 else Vehicle[5]
        # the angle of the Rectangle is under right-handed coordinate system, but the yaw angle in carla is left-handed coordinate system
        direction = [np.cos(angle_rad) * speed, - np.sin(angle_rad) * speed]
        arrow = Arrow(x=Vehicle[0], y=Vehicle[1], dx=direction[0], dy=direction[1], width=1.5)
        ax.add_patch(rectangle)
        ax.add_patch(arrow)
    # plot all the arrows
    arrow_xs = np.linspace(x_range[0] + 8, x_range[1] - 8, 3)
    arrow_ys = np.linspace(y_range[0] + 4, y_range[1] - 4, 2)
    ego_angle_rad = ego_obs[0, 4]
    # the angle of the Rectangle is under right-handed coordinate system, but the yaw angle in carla is left-handed coordinate system
    arrow_dir = [np.cos(ego_angle_rad) * ego_speed, - np.sin(ego_angle_rad) * ego_speed]
    for arrow_x in arrow_xs:
        for arrow_y in arrow_ys:
            arrow = Arrow(x=arrow_x, y=arrow_y, dx=arrow_dir[0], dy=arrow_dir[1], width=1, color='k', alpha=0.2)
            ax.add_patch(arrow)

    ax.set_xlim([-24, 24])
    ax.set_ylim([-9.5, 9.5])
    ax.set_aspect('equal', adjustable='box')

    return ax


def plot_multi_feasibility_region(args):
    set_torch_variable(args.device)
    torch.set_num_threads(args.threads)
    seed = args.plot_seed
    set_seed(seed)
    args_dict = vars(args)

    # load feasibility config
    feasibility_config_path = osp.join(args.ROOT_DIR, 'safebench/feasibility/config', args.feasibility_cfg)
    feasibility_config = load_config(feasibility_config_path)
    feasibility_config.update(args_dict)

    actor_num = args_dict['actor_num']
    x_range = args_dict['x_range']
    y_range = args_dict['y_range']
    height = args_dict['height']
    width = args_dict['width']
    spatial_interval = args_dict['spatial_interval']
    scenario_map = args_dict['scenario_map']
    map_name = scenario_map.split('_')[-1]
    min_dis_threshold = args_dict['min_dis_threshold']

    # set the logger
    log_path = osp.join(args.ROOT_DIR, 'safebench/feasibility/train_log', 'min_dis_threshold_' + min_dis_threshold + '_seed' + str(args.seed), scenario_map)
    log_exp_name = scenario_map
    logger = Logger(log_path, log_exp_name)

    # init the feasibility policy
    feasibility_policy = FEASIBILITY_LIST[feasibility_config['type']](feasibility_config, logger=logger)
    feasibility_policy.load_model(map_name)

    matplotlib.rcParams['font.family'] = 'Times New Roman'

    # create figure
    fig, axs = plt.subplots(
        nrows=2, ncols=2,
        figsize=(10, 4.2),
        constrained_layout=True,
    )
    ax1, ax2, ax3, ax4 = axs.flatten()

    my_x_ticks = np.arange(-24, 24.01, 5)
    my_y_ticks = np.arange(-9.5, 9.51, 5)

    labels = ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_xticklabels() + ax2.get_yticklabels() \
             + ax3.get_xticklabels() + ax3.get_yticklabels() + ax4.get_xticklabels() + ax4.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    '''
    subplot 1,2,3,4 : plot the feasible region and the learned feasible region for different v and theta
    '''
    # figure 1, ego at low speed (2m/s), BVs driving straight
    ego_obs1 = generate_ego_obs(ego_speed=4, yaw_angle=(0.0, 0.0), actor_num=actor_num, x_range=x_range, y_range=y_range, width=width, height=height, speed_range=(3, 3))
    ax1 = plot_feasibility_region(ax1, feasibility_policy, ego_obs1, ego_speed=2, spatial_interval=spatial_interval, actor_num=actor_num)
    # figure 2, ego at high speed (6m/s), BVs driving straight
    ax2 = plot_feasibility_region(ax2, feasibility_policy, ego_obs1, ego_speed=6, spatial_interval=spatial_interval, actor_num=actor_num)
    # figure 3, ego at normal speed (4m/s), BVs driving toward ego
    ego_obs2 = generate_ego_obs(ego_speed=4, yaw_angle=(-3.14/2, 3.14/2), actor_num=actor_num, x_range=x_range, y_range=y_range, width=width, height=height, speed_range=(3, 3))
    ax3 = plot_feasibility_region(ax3, feasibility_policy, ego_obs2, ego_speed=4, spatial_interval=spatial_interval, actor_num=actor_num)
    # figure 4, ego at normal speed (4m/s), BVs driving across under different directions
    ego_obs3 = generate_ego_obs(ego_speed=4, yaw_angle=(-3.14/6, 3.14/6), actor_num=actor_num, x_range=x_range, y_range=y_range, width=width, height=height, speed_range=(3, 3))
    ax4 = plot_feasibility_region(ax4, feasibility_policy, ego_obs3, ego_speed=4, spatial_interval=spatial_interval, actor_num=actor_num)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks(my_x_ticks)
        ax.set_yticks(my_y_ticks)
        ax.set_xlim((-24, 24))
        ax.set_ylim((-9.5, 9.5))
        ax.tick_params(labelsize=18)
        ax.set_xlim([-24, 24])
        ax.set_ylim([-9.5, 9.5])
        ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['right'].set_linewidth(0.5)
        ax.spines['top'].set_linewidth(0.5)
        ax.spines['top'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')

    save_dir = osp.join(args.ROOT_DIR, f'safebench/feasibility/figures/feasibility_region_{args.min_dis_threshold}.png')
    plt.savefig(save_dir, dpi=600)
    plt.show()


def read_df(df_path, ROOT_DIR):
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        print('>> ' + '-' * 20, 'Reading learning curve data', '-' * 20)

    else:
        dataframes = []
        pattern = r"run-min_dis_threshold_([0-9.]+)_seed(\d+)_Scenario(\d+)_Town(\d+)-tag-(.+)\.csv"
        root_dir = osp.join(ROOT_DIR, 'safebench/feasibility/processed_data/')
        # walk through all the dir in the root directory
        for root, dirs, files in os.walk(root_dir):
            file_names = [f for f in files if f.startswith('run-')]
            for file_name in file_names:
                match = re.search(pattern, file_name)
                if match:
                    min_dis_threshold = float(match.group(1))
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
                        'min_dis_threshold': np.repeat(min_dis_threshold, len(data)),
                        'scene': np.repeat(scenario, len(data)),
                        'label': np.repeat(label, len(data))
                }))
                else:
                    print("no matching")

        # merge all the dataframes
        df = pd.concat(dataframes, ignore_index=True)
        df.to_csv(osp.join(args.ROOT_DIR, 'safebench/feasibility/processed_data/learning_curve.csv'), index=False)
        print('>> ' + '-' * 20, 'Saving learning curve data', '-' * 20)

    return df


def format_label(label):
    parts = label.split('_')
    return f"${{{parts[1][0]}}}_{{{parts[1][1:]}}}$ {parts[2]}"


def plot_learning_curve(args):
    # read the learning curve data
    df_path = osp.join(args.ROOT_DIR, 'safebench/feasibility/processed_data/learning_curve.csv')
    all_df = read_df(df_path, args.ROOT_DIR)
    sns.set(style="darkgrid")
    labels = all_df['label'].drop_duplicates()

    for label in labels:
        # Filter dataframe for current label
        df = all_df[all_df['label'] == label]

        # create subplot
        scenes = df['scene'].drop_duplicates()
        subplots_height = 5
        aspect_ratio = 1.2
        figsize = (subplots_height * aspect_ratio, subplots_height)
        plt.rcParams['font.family'] = 'Times New Roman'
        fig, ax = plt.subplots(figsize=figsize)  # make sure axs are always 2D

        for scene in scenes:
            scene_df = df[df['scene'] == scene]

            # handel algorithm in each scene
            for algorithm, alg_df in scene_df.groupby('min_dis_threshold'):
                smoothed_dfs = []
                for seed, seed_df in alg_df.groupby('seed'):
                    seed_df = seed_df.sort_values('step')
                    seed_df['smoothed_value'] = smooth(seed_df['value'].values, sm=200)
                    smoothed_dfs.append(seed_df)

                # merge the smoothed data
                smoothed_df = pd.concat(smoothed_dfs)
                each_label = f'{scene}  min distance:{algorithm} m'
                # plot the mean value and the trust region
                sns.lineplot(ax=ax, data=smoothed_df, x='step', y='smoothed_value',
                             estimator='mean', errorbar=('ci', 95), label=each_label,
                             err_kws={"alpha": 0.2, "linewidth": 0.1})  # error_kws: the parameter for the trust region

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles[0:], labels=labels[0:], title="Collision_distance_threshold", loc="best", fontsize=8, title_fontsize=10)

            ax.set_xlabel('Step')
            ax.set_ylabel(format_label(label))

        save_dir = osp.join(args.ROOT_DIR, f'safebench/feasibility/figures/{label}_learning_curve.png')
        plt.savefig(save_dir, dpi=600)
        plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_route', type=str, default='safebench/feasibility/data')
    parser.add_argument('--scenario_map', '-sm', type=str, default='Scenario9_Town05')
    parser.add_argument('--feasibility_cfg', nargs='*', type=str, default='HJR.yaml')
    parser.add_argument('--data_filename', type=str, default='merged_data.hdf5')
    parser.add_argument('--min_dis_threshold', '-dis', type=str, default='0.1')
    parser.add_argument('--mode', '-m', type=str, default='region', choices=['region', 'data', 'learning_curve'])
    parser.add_argument('--actor_num', type=int, default=3)
    parser.add_argument('--spatial_interval', type=int, default=32)
    parser.add_argument('--x_range', type=tuple, default=(-22, 22))
    parser.add_argument('--y_range', type=tuple, default=(-7, 7))
    parser.add_argument('--width', type=float, default=4.9)
    parser.add_argument('--height', type=float, default=2.12)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--plot_seed', '-ps', type=int, default=77)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))))
    args = parser.parse_args()

    # 1. plot the feasibility data distribution
    if args.mode == 'data':
        plot_feasibility_data_distribution(args)

    # 2. plot the well-trained feasibility region
    if args.mode == 'region':
        plot_multi_feasibility_region(args)

    # 3. plot the learning curve of the feasibility
    if args.mode == 'learning_curve':
        plot_learning_curve(args)


