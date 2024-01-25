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

    # x, y position
    x_coords = obs[:, 0:-1, 0].flatten()
    y_coords = obs[:, 0:-1, 1].flatten()

    # relative yaw, speed
    yaw = obs[:, 0:-1, 4].flatten()
    speed = obs[:, 0:-1, 5].flatten()

    # throttle and speed
    throttle = action[:, 0]
    steering_angle = action[:, 1]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].scatter(x_coords, y_coords, alpha=0.7)
    axs[0, 0].set_title('Scatter Plot of Position Coordinates (x, y)')
    axs[0, 0].set_xlabel('X Coordinate')
    axs[0, 0].set_ylabel('Y Coordinate')

    axs[0, 1].hist(constraint_h, bins=30, color='orange', alpha=0.7)
    axs[0, 1].set_title('Distribution of Constraint h')
    axs[0, 1].set_xlabel('Constraint h')
    axs[0, 1].set_ylabel('Frequency')

    axs[1, 0].scatter(yaw, speed, alpha=0.7)
    axs[1, 0].set_title('Scatter Plot of Relative yaw and speed')
    axs[1, 0].set_xlabel('Relative yaw')
    axs[1, 0].set_ylabel('Speed')

    axs[1, 1].scatter(throttle, steering_angle, alpha=0.7)
    axs[1, 1].set_title('Scatter Plot of Throttle and Speed')
    axs[1, 1].set_xlabel('Throttle')
    axs[1, 1].set_ylabel('Steering angle')

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()
