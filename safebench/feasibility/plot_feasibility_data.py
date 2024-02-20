#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：plot_feasibility_data.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/1/25
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import colors
import os.path as osp
import torch

from safebench.feasibility import FEASIBILITY_LIST
from safebench.feasibility.dataset import OffRLDataset
from safebench.gym_carla.envs.utils import linear_map
from safebench.util.logger import Logger
from safebench.util.run_util import load_config
from safebench.util.torch_util import set_torch_variable, set_seed, CUDA, CPU


def plot_feasibility_data_distribution(args):
    # the route of the data need to be processed
    data_file_path = osp.join(args.ROOT_DIR, args.data_route, args.scenario_map, args.data_filename)
    dataset = OffRLDataset(data_file_path, device='CPU')

    obs = dataset.dataset_dict['obs']
    action = dataset.dataset_dict['actions']
    ego_min_dis = dataset.dataset_dict['ego_min_dis']
    ego_collision = dataset.dataset_dict['ego_collide']
    ego_collision_percentage = np.mean(ego_collision == 1.0) * 100

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

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    _, _, _, x_y_img = axs[0, 0].hist2d(non_zero_x, non_zero_y, bins=60, cmap='Blues', norm=LogNorm())
    axs[0, 0].set_title('Closest Vehicle Position (x, y)', fontsize=12)
    axs[0, 0].set_xlabel('X Coordinate')
    axs[0, 0].set_ylabel('Y Coordinate')
    x_y_bar = fig.colorbar(x_y_img, ax=axs[0, 0], label="Density")

    axs[0, 1].hist(ego_min_dis, bins=50, color='darkblue', alpha=0.9)
    axs[0, 1].set_title('Closest dis between Ego and BVs', fontsize=12)
    axs[0, 1].set_xlabel('Closest distance')
    text = 'Ego Collision: {:.2f}%'.format(ego_collision_percentage)
    axs[0, 1].text(0.5, 0.5, text, ha='center', va='center', fontsize=10, color='red', weight='bold')
    axs[0, 1].set_ylabel('Frequency')

    _, _, _, yaw_speed_img = axs[1, 0].hist2d(non_zero_yaw, non_zero_speed, bins=60, cmap='Blues', norm=LogNorm())
    axs[1, 0].set_title('Closest Vehicle yaw and Speed', fontsize=12)
    axs[1, 0].set_xlabel('Relative yaw')
    axs[1, 0].set_ylabel('Speed')
    yaw_speed_bar = fig.colorbar(yaw_speed_img, ax=axs[1, 0], label="Density")

    _, _, _, throttle_steering_angle_img = axs[1, 1].hist2d(throttle, steering_angle, bins=45, cmap='Blues', norm=LogNorm())
    axs[1, 1].set_title('Ego Vehicle Throttle and Speed', fontsize=12)
    axs[1, 1].set_xlabel('Throttle')
    axs[1, 1].set_ylabel('Steering angle')
    throttle_steering_angle_bar = fig.colorbar(throttle_steering_angle_img, ax=axs[1, 1], label="Density")

    plt.tight_layout()

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


def generate_ego_obs(ego_speed, yaw_angle=(-0.0, 0.0), actor_num=3, x_range=(-23, 23), y_range=(-9, 9), width=4.9, height=2.12, speed_range=(1, 5)):
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


def plot_feasibility_region(ax, agent, ego_obs, spatial_interval=10, ego_speed=6, actor_num=3, x_range=(-25, 25), y_range=(-10, 10), width=4.9, height=2.12):
    from matplotlib.patches import Rectangle, Arrow

    # the fake ego x, y coordinates
    x = np.linspace(x_range[0] + width/2, x_range[1] - width/2, spatial_interval)
    y = np.linspace(y_range[0] + height/2, y_range[1] - height / 2, spatial_interval)
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
    feasibility_value = agent.get_feasibility_value_from_state(batch_ego_obs)
    array_value = np.asarray(feasibility_value).reshape(x_grid.shape)

    norm = colors.Normalize(vmin=-8.5, vmax=4)

    ct = ax.contourf(
        x_grid, y_grid, array_value,
        norm=norm,
        levels=15,
        cmap='rainbow',
        alpha=0.5
    )
    try:
        ct_line = ax.contour(
            x_grid, y_grid, array_value,
            levels=[0], colors='#32ABD6',
            linewidths=2.0, linestyles='solid'
        )
        ax.clabel(ct_line, inline=True, fontsize=10, fmt=r'0', )
    except ValueError:
        pass

    cb = plt.colorbar(ct, ax=ax, shrink=0.8, pad=0.01)
    cb.ax.tick_params(labelsize=7)

    # plot all the vehicles
    for i, Vehicle in enumerate(ego_obs):
        # the x,y in Rectangle in the bottom left corner
        color = 'r' if i == 0 else (0.30, 0.52, 0.74)
        rectangle = Rectangle((Vehicle[0]-Vehicle[2] / 2, Vehicle[1]-Vehicle[3] / 2), width=Vehicle[2], height=Vehicle[3], angle=np.degrees(Vehicle[4]), rotation_point='center', fill=True, alpha=0.5, color=color)
        angle_rad = Vehicle[4]
        speed = ego_speed if i == 0 else Vehicle[5]
        direction = [np.cos(angle_rad) * speed, np.sin(angle_rad) * speed]
        arrow = Arrow(x=Vehicle[0], y=Vehicle[1], dx=direction[0], dy=direction[1], width=1.5)
        ax.add_patch(rectangle)
        ax.add_patch(arrow)

    ax.set_xlim([-24, 24])
    ax.set_ylim([-9.5, 9.5])
    ax.set_aspect('equal', adjustable='box')

    return ax


def plot_multi_feasibility_region(args):
    set_torch_variable(args.device)
    torch.set_num_threads(args.threads)
    seed = args.seed
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
    map_name = args_dict['map_name']
    scenario_map = args_dict['scenario_map']

    # set the logger
    log_path = osp.join(args.ROOT_DIR, 'safebench/feasibility/train_log', scenario_map)
    log_exp_name = scenario_map
    logger = Logger(log_path, log_exp_name)

    # init the feasibility policy
    feasibility_policy = FEASIBILITY_LIST[feasibility_config['type']](feasibility_config, logger=logger)
    feasibility_policy.load_model(map_name)

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
    ego_obs1 = generate_ego_obs(ego_speed=4, yaw_angle=(0.0, 0.0), actor_num=actor_num, x_range=x_range, y_range=y_range, width=width, height=height)
    ax1 = plot_feasibility_region(ax1, feasibility_policy, ego_obs1, spatial_interval=spatial_interval, ego_speed=4, actor_num=actor_num)

    ego_obs2 = generate_ego_obs(ego_speed=6, yaw_angle=(0.0, 0.0), actor_num=actor_num, x_range=x_range, y_range=y_range, width=width, height=height)
    ax2 = plot_feasibility_region(ax2, feasibility_policy, ego_obs2, spatial_interval=spatial_interval, ego_speed=6, actor_num=actor_num)

    ego_obs3 = generate_ego_obs(ego_speed=4, yaw_angle=(-3.14/2, -3.14/2), actor_num=actor_num, x_range=x_range, y_range=y_range, width=width, height=height)
    ax3 = plot_feasibility_region(ax3, feasibility_policy, ego_obs3, spatial_interval=spatial_interval, ego_speed=4, actor_num=actor_num)

    ego_obs4 = generate_ego_obs(ego_speed=4, yaw_angle=(3.14/2, 3.14/2), actor_num=actor_num, x_range=x_range, y_range=y_range, width=width, height=height)
    ax4 = plot_feasibility_region(ax4, feasibility_policy, ego_obs4, spatial_interval=spatial_interval, ego_speed=4, actor_num=actor_num)

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

    plt.show()
    # plt.savefig(f"/imgs/viz_map.png", dpi=600)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_route', type=str, default='safebench/feasibility/data')
    parser.add_argument('--scenario_map', type=str, default='Scenario9_Town05')
    parser.add_argument('--feasibility_cfg', nargs='*', type=str, default='HJR.yaml')
    parser.add_argument('--map_name', '-map', type=str, default='Town05')
    parser.add_argument('--data_filename', type=str, default='merged_data.hdf5')
    parser.add_argument('--plot_data', '-data', action='store_true', help='plot offline data distribution')
    parser.add_argument('--plot_feasibility_region', '-region', action='store_true', help='plot feasibility region')
    parser.add_argument('--actor_num', type=int, default=3)
    parser.add_argument('--spatial_interval', type=int, default=32)
    parser.add_argument('--x_range', type=tuple, default=(-22, 22))
    parser.add_argument('--y_range', type=tuple, default=(-7, 7))
    parser.add_argument('--width', type=float, default=4.9)
    parser.add_argument('--height', type=float, default=2.12)
    parser.add_argument('--seed', '-s', type=int, default=6)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))))
    args = parser.parse_args()

    # 1. plot the feasibility data distribution
    if args.plot_data:
        plot_feasibility_data_distribution(args)

    # 2. plot the well-trained feasibility region
    if args.plot_feasibility_region:
        plot_multi_feasibility_region(args)



