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
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import h5py
import os.path as osp
import torch
from safebench.feasibility.dataset import OffRLDataset


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_route', type=str, default='feasibility/data')
    parser.add_argument('--scenario_map', type=str, default='scenario9_Town05')
    parser.add_argument('--data_filename', type=str, default='merged_data.hdf5')
    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__)))))
    args = parser.parse_args()

    # the route of the data need to be processed
    data_file_path = osp.join(args.ROOT_DIR, args.data_route, args.scenario_map, args.data_filename)
    dataset = OffRLDataset(data_file_path, device=args.device)

    obs = dataset.dataset_dict['obs']
    action = dataset.dataset_dict['actions']
    constraint_h = dataset.dataset_dict['constraint_h']

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

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    _, _, _, x_y_img = axs[0, 0].hist2d(non_zero_x, non_zero_y, bins=60, cmap='Blues', norm=LogNorm())
    axs[0, 0].set_title('Distribution of the Closest Vehicle Position Coordinates (x, y)')
    axs[0, 0].set_xlabel('X Coordinate')
    axs[0, 0].set_ylabel('Y Coordinate')
    x_y_bar = fig.colorbar(x_y_img, ax=axs[0, 0], label="Density")

    axs[0, 1].hist(constraint_h, bins=50, color='darkblue', alpha=0.9)
    axs[0, 1].set_title('Distribution of Constraint h')
    axs[0, 1].set_xlabel('Constraint h')
    axs[0, 1].set_ylabel('Frequency')

    _, _, _, yaw_speed_img = axs[1, 0].hist2d(non_zero_yaw, non_zero_speed, bins=60, cmap='Blues', norm=LogNorm())
    axs[1, 0].set_title('Distribution of the Closest Vehicle Relative yaw and speed')
    axs[1, 0].set_xlabel('Relative yaw')
    axs[1, 0].set_ylabel('Speed')
    yaw_speed_bar = fig.colorbar(yaw_speed_img, ax=axs[1, 0], label="Density")

    _, _, _, throttle_steering_angle_img = axs[1, 1].hist2d(throttle, steering_angle, bins=45, cmap='Blues', norm=LogNorm())
    axs[1, 1].set_title('Distribution of Ego Vehicle Throttle and Speed')
    axs[1, 1].set_xlabel('Throttle')
    axs[1, 1].set_ylabel('Steering angle')
    throttle_steering_angle_bar = fig.colorbar(throttle_steering_angle_img, ax=axs[1, 1], label="Density")

    plt.tight_layout()

    plt.show()
