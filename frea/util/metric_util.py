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

from copy import deepcopy
import argparse

from frea.scenario.scenario_definition.atomic_criteria import Status


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
    avg_yaw_velocity = total_yaw_change / delta_time if abs(delta_time) > 0.001 else 0  # prevent the delta_time is 0

    return avg_yaw_velocity


def get_route_scores(record_dict, use_feasibility, scenario_agent_learnable, time_out=30):
    # safety level
    sum_out_of_road_length = 0

    for data_id, sequence in record_dict.items():  # for each data id (small scenario)
        sum_out_of_road_length += cal_out_of_road_length(sequence)
    out_of_road_length = sum_out_of_road_length / len(record_dict)

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
        avg_sequence_acc = 0
        for time_stamp in sequence:
            avg_sequence_acc += math.sqrt(time_stamp['ego_acc'][0] ** 2 + time_stamp['ego_acc'][1] ** 2)
        total_acc += avg_sequence_acc / len(sequence)
        total_yaw_velocity += cal_avg_yaw_velocity(sequence)
    avg_acc = total_acc / len(record_dict)
    avg_yaw_velocity = total_yaw_velocity / len(record_dict)

    predefined_max_values = {
        # safety level
        'out_of_road_length': 1,

        # task performance level
        'distance_to_route': 1,
        'incomplete_route': 1,
        'running_time': time_out,

        # comfort level
        'average acceleration': 5,
        'average yaw velocity': 0.6,
    }

    weights = {
        # safety level
        'out_of_road_length': 0.1,

        # task performance level
        'distance_to_route': 0.1,
        'incomplete_route': 0.3,
        'running_time': 0.1,

        # comfort level
        'average acceleration': 0.2,
        'average yaw velocity': 0.2,
    }

    scores = {
        # safety level
        'out_of_road_length': out_of_road_length,

        # task performance level
        'distance_to_route': avg_distance_to_route,
        'incomplete_route': 1 - route_completion,
        'running_time': avg_time_spent,

        # comfort level
        'average acceleration': avg_acc,
        'average yaw velocity': avg_yaw_velocity,
    }

    all_scores = {key: round(max(0, min(value / predefined_max_values[key], 1)), 2) for key, value in scores.items()}
    final_score = 0
    for key, score in all_scores.items():
        final_score += score * weights[key]
    all_scores['final_score'] = round(1 - final_score, 2)  # the score of ego's driving behavior

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
