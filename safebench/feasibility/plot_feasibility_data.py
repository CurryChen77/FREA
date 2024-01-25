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

    axs[0, 0].scatter(non_zero_x, non_zero_y, alpha=0.7)
    axs[0, 0].set_title('Scatter Plot of the Closest Vehicle Position Coordinates (x, y)')
    axs[0, 0].set_xlabel('X Coordinate')
    axs[0, 0].set_ylabel('Y Coordinate')

    axs[0, 1].hist(constraint_h, bins=30, color='orange', alpha=0.7)
    axs[0, 1].set_title('Distribution of Constraint h')
    axs[0, 1].set_xlabel('Constraint h')
    axs[0, 1].set_ylabel('Frequency')

    axs[1, 0].scatter(non_zero_yaw, non_zero_speed, alpha=0.7)
    axs[1, 0].set_title('Scatter Plot of the Closest Vehicle Relative yaw and speed')
    axs[1, 0].set_xlabel('Relative yaw')
    axs[1, 0].set_ylabel('Speed')

    axs[1, 1].scatter(throttle, steering_angle, alpha=0.7)
    axs[1, 1].set_title('Scatter Plot of Ego Vehicle Throttle and Speed')
    axs[1, 1].set_xlabel('Throttle')
    axs[1, 1].set_ylabel('Steering angle')

    plt.tight_layout()

    plt.show()
