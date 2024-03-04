#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：convert_json.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/11/5
"""

import json

# create an empty list
data = []
data_id = 0
# create the key-value pairs
for scenario_id in range(9, 9+1):
    for route_id in range(8, 27+1):
        if route_id == 17 or route_id == 22:
            pass  # remove the route in Town04
        else:
            for i in range(5):  # repeating times
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
json_file_name = "../safebench/scenario/config/scenario_type/carla_scenario_9_Town5-2-1_5x.json"

# open the json file and write in the data
with open(json_file_name, 'w') as json_file:
    # use json.dump() to write in the data and set the indent
    json.dump(data, json_file, indent=4)

print(f"successfully creating the JSON file {json_file_name}")
