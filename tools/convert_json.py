#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：convert_json.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/11/5
"""
import argparse
import json

town_dict = {
    'Town01': [9, 11, 19],
    'Town02': [12, 13, 14, 15, 16, 20, 21],
    'Town04': [17, 22],
    'Town05': [8, 10, 18, 23, 24, 25, 26, 27]
}


def main(args):
    town_list = []
    town_names = args.town_names
    repeat_times = args.repeat_times
    print("Selecting Town:", town_names)
    for town in town_names:
        town_list.extend(town_dict[town])
    # create an empty list
    data = []
    data_id = 0
    # create the key-value pairs
    for scenario_id in range(9, 9+1):
        for route_id in town_list:
            for i in range(repeat_times):  # repeating times
                item = {
                    "data_id": data_id,
                    "scenario_id": int(scenario_id),
                    "route_id": int(route_id),
                    "risk_level": None,
                    "parameters": None
                }
                data.append(item)
                data_id += 1

    # set the file name path
    town_path = "_".join(town_names)
    json_file_name = f"../safebench/scenario/config/scenario_type/Scenario9_{town_path}_{repeat_times}x.json"

    # open the json file and write in the data
    with open(json_file_name, 'w') as json_file:
        # use json.dump() to write in the data and set the indent
        json.dump(data, json_file, indent=4)

    print(f"successfully creating the JSON file {json_file_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--town_names', nargs='+', default=['Town05', 'Town02'])
    parser.add_argument('--repeat_times', '-r', type=int, default=10)
    args = parser.parse_args()
    main(args)