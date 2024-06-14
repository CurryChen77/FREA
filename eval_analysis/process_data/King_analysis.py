#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：King_analysis.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/6/12
"""
import os
import os.path as osp
import json
import numpy as np
import torch
from tqdm import tqdm
from frea.util.torch_util import CUDA, CPU
from frea.feasibility import FEASIBILITY_LIST
from frea.util.logger import Logger
from frea.util.run_util import load_config
from distance3d import gjk, colliders


def compute_R(yaw):
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return R


def compute_box2origin_2D(vehicle_location, vehicle_yaw):
    t = np.array([
        vehicle_location[0],
        vehicle_location[1],
        0
    ])

    r = compute_R(vehicle_yaw)

    size = np.array([4.4, 1.8, 0])

    box2origin = np.eye(4)
    box2origin[:3, :3] = r
    box2origin[:3, 3] = t

    return box2origin, size


def get_min_distance_across_centers(coords_1, coords_2):
    """
    Get the distance between two point coords -> [x, y]
    """
    return np.linalg.norm(coords_1 - coords_2)


def get_min_distance_across_boxes(coords_1, coords_2, yaw1, yaw2):
    box2origin_1, size_1 = compute_box2origin_2D(coords_1, yaw1)
    box2origin_2, size_2 = compute_box2origin_2D(coords_2, yaw2)
    box_collider_1 = colliders.Box(box2origin_1, size_1)
    box_collider_2 = colliders.Box(box2origin_2, size_2)
    dist, closest_point_box, closest_point_box2, _ = gjk.gjk(
        box_collider_1, box_collider_2)
    return dist


def get_ego_BV_min_distance(ego_obs, yaw_list):
    ego_coords = ego_obs[0, :2]  # Assuming the ego coordinates are the first row, and we need first two columns (x, y)
    ego_yaw = yaw_list[0]

    min_dis = np.inf  # Initialize with infinity to find minimum distance

    for i in range(1, len(ego_obs)):  # Start from 1 to skip ego itself
        BV_coords = ego_obs[i, :2]  # Assuming BV coordinates are in the same format
        BV_yaw = yaw_list[i]
        distance = get_min_distance_across_boxes(ego_coords, BV_coords, ego_yaw, BV_yaw)
        if distance < min_dis:
            min_dis = distance

    return min_dis


def get_feasibility_policy(feasibility_config, ROOT_DIR):
    logger_path = osp.join(ROOT_DIR, 'eval_analysis/King_result')
    logger = Logger(logger_path, all_map_name=['Town05, Town02'])
    feasibility_policy = FEASIBILITY_LIST[feasibility_config['type']](feasibility_config, logger=logger)
    feasibility_policy.load_model(map_name='Town05')
    print(">> Successfully loading the feasibility policy")
    return feasibility_policy


def create_yaw_rotation_matrix(yaw):
    """
    Create a 3x3 rotation matrix for a given yaw angle (in radians).
    """
    cos_theta = np.cos(yaw)
    sin_theta = np.sin(yaw)

    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])

    return rotation_matrix


def get_relative_transform(ego_coords, vehicle_coords, ego_yaw):
    """
    return the relative transform from ego_pose to vehicle pose
    """
    relative_pos = np.array(vehicle_coords + [0]) - np.array(ego_coords + [0])
    rot = create_yaw_rotation_matrix(ego_yaw).T
    relative_pos = rot @ relative_pos

    # transform to the right-handed system
    relative_pos[1] = - relative_pos[1]

    return relative_pos


def get_forward_speed_2d(yaw, velocity):
    """
    Convert the vehicle yaw directly to forward speed in 2D
    the velocity from King is []
    """
    vel_np = np.array([velocity[0], velocity[1]])
    orientation = np.array([np.cos(yaw), np.sin(yaw)])
    speed = np.dot(vel_np, orientation)
    return speed


def normalize_angle(x):
    x = x % (2 * np.pi)  # force in range [0, 2 pi)
    if x > np.pi:  # move to [-pi, pi)
        x -= 2 * np.pi
    return x


def get_relative_info(ego_coords, vehicle_coords, ego_yaw, vehicle_yaw, vehicle_velocity):
    """
        get the relative actor info from the view of center vehicle
        info [x, y, bbox_x, bbox_y, yaw, forward speed]
    """
    # relative yaw angle
    relative_yaw = normalize_angle(vehicle_yaw - ego_yaw)
    # relative pos
    relative_pos = get_relative_transform(ego_coords=ego_coords, vehicle_coords=vehicle_coords, ego_yaw=ego_yaw)

    vehicle_forward_vel = get_forward_speed_2d(yaw=vehicle_yaw, velocity=vehicle_velocity)  # In m/s
    # the extent x and y from King setting
    vehicle_info = [relative_pos[0], relative_pos[1], 4.4, 1.8, relative_yaw, vehicle_forward_vel]

    return vehicle_info


def sort_vehicle_infos_by_distance(ego_info, vehicle_infos):
    ego_coords, ego_yaw, ego_velocity = ego_info
    sorted_vehicle_infos = sorted(
        vehicle_infos,
        key=lambda vehicle_info: get_min_distance_across_centers(np.array(ego_coords), np.array(vehicle_info[0]))
    )

    return sorted_vehicle_infos


def form_ego_obs(ego_info, vehicle_infos, desired_nearby_vehicle=3):
    ego_obs = []
    abs_yaw = []
    ego_coords, ego_yaw, ego_velocity = ego_info
    ego_rel_info = get_relative_info(ego_coords=ego_coords, vehicle_coords=ego_coords, ego_yaw=ego_yaw, vehicle_yaw=ego_yaw, vehicle_velocity=ego_velocity)
    ego_obs.append(ego_rel_info)
    abs_yaw.append(ego_yaw)
    vehicle_infos = sort_vehicle_infos_by_distance(ego_info, vehicle_infos)
    for vehicle_info in vehicle_infos:
        if len(ego_obs) < desired_nearby_vehicle:
            vehicle_coords, vehicle_yaw, vehicle_velocity = vehicle_info
            vehicle_rel_info = get_relative_info(ego_coords=ego_coords, vehicle_coords=vehicle_coords, ego_yaw=ego_yaw, vehicle_yaw=vehicle_yaw, vehicle_velocity=vehicle_velocity)
            ego_obs.append(vehicle_rel_info)
            abs_yaw.append(vehicle_yaw)
        else:
            break
    while len(ego_obs) < desired_nearby_vehicle:
        ego_obs.append([0] * len(ego_rel_info))
        abs_yaw.append(0)

    return np.array(ego_obs, dtype=np.float32), np.array(abs_yaw, dtype=np.float32)


def find_negative_transition(fea_V):
    # find the negative situation
    mask = (fea_V < 0) & (torch.roll(fea_V, shifts=-1) < 0)
    # find the first index
    start_index = torch.argmax(mask.int()).item()
    return start_index if mask.any() else None


def get_feasibility_metric(collision_fea_dis, feasibility_policy):
    fea_boundary_dis = []
    feasibility_Vs_list = []
    for fea_dis in collision_fea_dis.values():
        fea_obs = CUDA(torch.stack([torch.FloatTensor(row[0]) for row in fea_dis]))
        feasibility_Vs = feasibility_policy.get_feasibility_Vs(fea_obs)
        feasibility_Vs_list.append(feasibility_Vs)
        reversed_fea_V = torch.flip(feasibility_Vs, dims=[0])
        negative_index = find_negative_transition(reversed_fea_V)
        if negative_index:
            distance = torch.tensor([row[1] for row in fea_dis])
            reversed_distance = torch.flip(distance, dims=[0])
            fea_boundary_dis.append(reversed_distance[negative_index].item())
    infeasible_ratio = (torch.stack(feasibility_Vs_list, dim=0) > 0).float().mean().item()
    return fea_boundary_dis, infeasible_ratio


def get_collision_severity(all_sequence, all_file_abs_yaw_list, feasibility_policy):
    collision_fea_dis = {}
    for i, sequence in enumerate(all_sequence):
        collision_fea_dis[i] = []
        for j, ego_obs_state in enumerate(sequence):
            ego_BV_min_dis = get_ego_BV_min_distance(ego_obs_state, all_file_abs_yaw_list[i][j])
            collision_fea_dis[i].append([ego_obs_state, ego_BV_min_dis])

    fea_boundary_dis, infeasible_ratio = get_feasibility_metric(collision_fea_dis, feasibility_policy)

    return fea_boundary_dis, infeasible_ratio


def main(args_dict):
    ROOT_DIR = args_dict['ROOT_DIR']
    feasibility_cfg_path = osp.join(ROOT_DIR, args_dict['feasibility_cfg_path'])
    feasibility_config = load_config(feasibility_cfg_path)
    feasibility_config.update(args_dict)
    feasibility_policy = get_feasibility_policy(feasibility_config, ROOT_DIR)

    base_dir = "eval_analysis/King_result"
    collision_count = 0
    total_count = 0
    all_file_ego_obs_list = []
    all_file_abs_yaw_list = []

    for folder in tqdm(os.listdir(base_dir), desc="Processing folders"):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            record_file_path = os.path.join(folder_path, os.path.basename(folder_path), 'scenario_records.json')
            result_file_path = os.path.join(folder_path, os.path.basename(folder_path), 'results.json')

            if os.path.exists(record_file_path) and os.path.exists(result_file_path):
                with open(result_file_path, 'r') as file:
                    result_data = json.load(file)
                    # only use the final iteration's result
                    max_iteration = result_data["first_metrics"]["iteration"]

                    num_agents = result_data["meta_data"]["Num_agents"]

                    with open(record_file_path, 'r') as file:
                        record_data = json.load(file)

                    # only evaluate the collision trajectory
                    for iteration in range(max_iteration):
                        total_count += 1
                        adv_collision = result_data["all_iterations"][str(iteration)]["adv_collision"]
                        if adv_collision:
                            collision_count += 1
                            sequences_data = record_data["states"][iteration]
                            ego_obs_list = []
                            abs_yaw_list = []
                            for step_state in sequences_data:
                                ego_info = [step_state['pos'][0], step_state['yaw'][0][0], step_state['vel'][0]]
                                vehicle_infos = []
                                for agent_index in range(1, num_agents):
                                    pos = step_state['pos'][agent_index]
                                    yaw = step_state['yaw'][agent_index][0]
                                    vel = step_state['vel'][agent_index]
                                    vehicle_infos.append([pos, yaw, vel])
                                ego_obs, abs_yaw = form_ego_obs(ego_info, vehicle_infos)
                                ego_obs_list.append(ego_obs)
                                abs_yaw_list.append(abs_yaw)
                            all_file_ego_obs_list.append(ego_obs_list)
                            all_file_abs_yaw_list.append(abs_yaw_list)

    # get the collision severity of the King trajectories
    fea_boundary_dis, infeasible_ratio = get_collision_severity(all_file_ego_obs_list, all_file_abs_yaw_list, feasibility_policy)
    print("Infeasible Distance (m):", np.mean(fea_boundary_dis))
    print("Infeasible Ratio (%):", round(infeasible_ratio * 100, 2))
    print(f"Collision Rate (%): ", round(collision_count/total_count * 100, 2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))))
    parser.add_argument('--feasibility_cfg_path', nargs='*', type=str, default='frea/feasibility/config/HJR.yaml')
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)

