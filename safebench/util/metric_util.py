#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：metric_util.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""
from collections import Counter

import joblib
import math
import numpy as np

from copy import deepcopy
import argparse

import torch
from safebench.scenario.scenario_definition.atomic_criteria import Status


def cal_out_of_road_length(sequence):
    out_of_road_raw = [i['off_road'] for i in sequence]
    out_of_road = deepcopy(out_of_road_raw)
    for i, out in enumerate(out_of_road_raw):
        if out and i + 1 < len(out_of_road_raw):
            out_of_road[i + 1] = True

    total_length = 0
    for i, out in enumerate(out_of_road):
        if i == 0:
            continue
        if out:
            total_length += sequence[i]['driven_distance'] - sequence[i - 1]['driven_distance']

    return total_length


def cal_avg_yaw_velocity(sequence):
    total_yaw_change = 0
    for i, time_stamp in enumerate(sequence):
        if i == 0:
            continue
        total_yaw_change += abs(sequence[i]['ego_yaw'] - sequence[i - 1]['ego_yaw'])
    total_yaw_change = total_yaw_change / 180 * math.pi
    delta_time = sequence[-1]['current_game_time'] - sequence[0]['current_game_time']
    avg_yaw_velocity = total_yaw_change / delta_time if delta_time == 0 else 0  # prevent the delta_time is 0

    return avg_yaw_velocity


def get_route_scores(record_dict, use_feasibility, scenario_agent_learnable, time_out=30):
    # safety level
    if scenario_agent_learnable:
        # learnable scenario agent
        num_collision = 0
        sum_out_of_road_length = 0
        collision_attacker = 0

        for data_id, sequence in record_dict.items():  # for each data id (small scenario)
            collision_count = Counter()
            all_count = Counter()

            for step in sequence:
                collision_count.update({k: 1 for k, v in step['CBVs_collision'].items() if v})
                all_count.update(step['CBVs_collision'].keys())

            num_collision += len(collision_count)
            collision_attacker += len(all_count)

            sum_out_of_road_length += cal_out_of_road_length(sequence)

        collision_rate = num_collision / collision_attacker
        out_of_road_length = sum_out_of_road_length / len(record_dict)
    else:
        # rule-based scenario agent
        num_collision = 0
        sum_out_of_road_length = 0
        for data_id, sequence in record_dict.items():  # for each data id (small scenario)
            for step in sequence:  # count all the collision event along the trajectory
                if step['collision'][0] == Status.FAILURE:
                    num_collision += 1
            sum_out_of_road_length += cal_out_of_road_length(sequence)
        collision_rate = num_collision / len(record_dict)
        out_of_road_length = sum_out_of_road_length / len(record_dict)

    # near miss
    num_near_step = 0
    total_step = 0
    for data_id, sequence in record_dict.items():  # for each data id (small scenario)
        for step in sequence:  # count all the collision event along the trajectory
            if step['ego_min_dis'] < 1:
                num_near_step += 1
            if step['ego_min_dis'] < 25:
                total_step += 1

    near_rate = num_near_step / total_step
    if near_rate == 0:
        near_miss_rate = 0
    else:
        near_miss_rate = 1.0 - (num_collision / num_near_step)

    # feasibility eval
    unavoidable_rate = 0
    if use_feasibility:
        num_unavoidable_collision = 0
        total_step = 0
        for data_id, sequence in record_dict.items():  # for each data id (small scenario)
            total_step += len(sequence)
            for step in sequence:  # count all the collision event along the trajectory
                if step['feasibility_V'] >= 0.0:
                    num_unavoidable_collision += 1

        unavoidable_rate = num_unavoidable_collision / total_step

    # task performance level
    total_route_completion = 0
    total_time_spent = 0
    total_distance_to_route = 0
    for data_id, sequence in record_dict.items():
        total_route_completion += sequence[-1]['route_complete'] / 100
        total_time_spent += sequence[-1]['current_game_time'] - sequence[0]['current_game_time']
        avg_distance_to_route = 0
        for time_stamp in sequence:
            avg_distance_to_route += time_stamp['distance_to_route']
        total_distance_to_route += avg_distance_to_route / len(sequence)

    avg_distance_to_route = total_distance_to_route / len(record_dict)
    route_completion = total_route_completion / len(record_dict)
    avg_time_spent = total_time_spent / len(record_dict)

    # comfort level
    num_lane_invasion = 0
    total_acc = 0
    total_yaw_velocity = 0
    for data_id, sequence in record_dict.items():
        num_lane_invasion += sequence[-1]['lane_invasion']
        avg_acc = 0
        for time_stamp in sequence:
            avg_acc += math.sqrt(time_stamp['ego_acceleration_x'] ** 2 + time_stamp['ego_acceleration_y'] ** 2 + time_stamp['ego_acceleration_z'] ** 2)
        total_acc += avg_acc / len(sequence)
        total_yaw_velocity += cal_avg_yaw_velocity(sequence)

    predefined_max_values = {
        # safety level
        'collision_rate': 1,
        'out_of_road_length': 10,
        'near_miss_rate': 1,
        'near_rate': 1,

        # task performance level
        'distance_to_route': 5,
        'incomplete_route': 1,
        'running_time': time_out,
    }

    weights = {
        # safety level
        'collision_rate': 0.2,
        'out_of_road_length': 0.05,
        'near_miss_rate': 0.3,
        'near_rate': 0.3,

        # task performance level
        'distance_to_route': 0.05,
        'incomplete_route': 0.05,
        'running_time': 0.05,
    }

    scores = {
        # safety level
        'collision_rate': collision_rate,
        'out_of_road_length': out_of_road_length,
        'near_miss_rate': near_miss_rate,
        'near_rate': near_rate,

        # task performance level
        'distance_to_route': avg_distance_to_route,
        'incomplete_route': 1 - route_completion,
        'running_time': avg_time_spent,
    }
    if use_feasibility:
        scores['unavoidable_rate'] = unavoidable_rate
        weights['unavoidable_rate'] = - 0.3
        weights['collision_rate'] = 0.1
        weights['near_rate'] = 0.1
        predefined_max_values['unavoidable_rate'] = 1

    all_scores = {key: round(value/predefined_max_values[key], 4) for key, value in scores.items()}
    final_score = 0
    for key, score in all_scores.items():
        final_score += score * weights[key]
    all_scores['final_score'] = round(final_score, 4)  # change from the lower, the better to the higher, the better

    return all_scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_file', default='/home/carla/output/testing_records/record.pkl')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()
    record = joblib.load(args.record_file)
    # all_scores, normalized_scores, final_score = get_scores(record)
    # print('overall score:', final_score)
