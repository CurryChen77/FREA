#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：observation_util.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/22
"""


import carla
from safebench.agent.agent_utils.coordinate_utils import normalize_angle
from safebench.agent.PlanT.dataset import split_large_BB, generate_batch
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
import numpy as np
from rdp import rdp

Leaderboard_base_sensor = [
    {
    'type': 'sensor.opendrive_map',
    'reading_frequency': 1e-6,
    'id': 'hd_map'
    },
    {
        'type': 'sensor.other.imu',
        'x': 0.0, 'y': 0.0, 'z': 0.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        'sensor_tick': 0.05,
        'id': 'imu'
    },
    {
        'type': 'sensor.other.gnss',
        'x': 0.0, 'y': 0.0, 'z': 0.0,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        'sensor_tick': 0.01,
        'id': 'gps'
    },
    {
        'type': 'sensor.speedometer',
        'reading_frequency': 20,
        'id': 'speed'
    }
]

Plant_sensor = Leaderboard_base_sensor + [
    {
        'type': 'sensor.camera.rgb',
        'x': -1.5,
        'y': 0.0,
        'z': 2.0,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': 0.0,
        'width': 900,
        'height': 256,
        'fov': 100,
        'id': 'rgb_front'
    },
    {
        'type': 'sensor.camera.rgb',
        'x': -1.5,
        'y': 0.0,
        'z': 2.0,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': 0.0,
        'width': 900,
        'height': 256,
        'fov': 100,
        'id': 'rgb_augmented'
    },
    {
        'type': 'sensor.lidar.ray_cast',
        'x': 1.3,
        'y': 0.0,
        'z': 2.5,
        'roll': 0,
        'pitch': 0,
        'yaw': -90.0,
        'rotation_frequency': 10,
        'points_per_second': 600000,
        'id': 'lidar'
    }
]

Safebench_sensor = [
    {
        'type': 'sensor.camera.rgb',
        'x': 0.8,
        'y': 0.0,
        'z': 1.7,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': 0.0,
        'width': 128,
        'height': 128,
        'fov': 110,
        'sensor_tick': 0.02,
        'id': 'rbg_front'
    },
    {
        'type': 'sensor.lidar.ray_cast',
        'x': 0.0,
        'y': 0.0,
        'z': 2.1,
        'roll': 0,
        'pitch': 0,
        'yaw': -90.0,
        'channels': 16,
        'range': 1000,
        'id': 'lidar'
    },
]

SENSOR_LIST = {
    'leaderboard_base': Leaderboard_base_sensor,
    'plant': Plant_sensor,
    'safebench': Safebench_sensor,
}


def get_forward_speed(transform, velocity):
    """ Convert the vehicle transform directly to forward speed """

    vel_np = np.array([velocity.x, velocity.y, velocity.z])
    pitch = np.deg2rad(transform.rotation.pitch)
    yaw = np.deg2rad(transform.rotation.yaw)
    orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    speed = np.dot(vel_np, orientation)
    return speed


def get_relative_transform(ego_matrix, vehicle_matrix):
    """
    return the relative transform from ego_pose to vehicle pose
    """
    relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
    rot = ego_matrix[:3, :3].T
    relative_pos = rot @ relative_pos

    # transform to right-handed system
    relative_pos[1] = - relative_pos[1]

    # transform relative pos to virtual lidar system
    rot = np.eye(3)
    trans = - np.array([1.3, 0.0, 2.5])
    relative_pos = rot @ relative_pos + trans

    return relative_pos


class AgentStateEncoder(object):
    def __init__(self, model_path):
        self.model_path = model_path

    def get_encoded_state(self, ego_vehicle, ego_nearby_vehicles, waypoints, traffic_light_hazard):
        if len(waypoints) > 1:
            target_point, _ = waypoints[1]  # the preview waypoint
        else:
            target_point, _ = waypoints[0]  # the first waypoint
        label_raw = self.get_bev_boxes(ego_vehicle, ego_nearby_vehicles, waypoints)
        input_batch = self.get_input_batch(label_raw, target_point, traffic_light_hazard)
        x, y, _, tp, light = input_batch
        _, _, pred_wp, attn_map = self.net(x, y, target_point=tp, light_hazard=light)
    def get_bev_boxes(self, ego_veh, ego_nearby_vehicles, waypoints):
        """
            modify from the PlanT
        """
        # -----------------------------------------------------------
        # Ego vehicle
        # -----------------------------------------------------------

        # add vehicle velocity and brake flag
        ego_transform = CarlaDataProvider.get_transform_after_tick(ego_veh)
        ego_control = ego_veh.get_control()
        ego_velocity = ego_veh.get_velocity()
        ego_speed = get_forward_speed(transform=ego_transform, velocity=ego_velocity)  # In m/s
        ego_brake = ego_control.brake
        ego_rotation = ego_transform.rotation
        ego_matrix = np.array(ego_transform.get_matrix())
        ego_extent = ego_veh.bounding_box.extent
        ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z]) * 2.
        ego_yaw = ego_rotation.yaw / 180 * np.pi
        relative_yaw = 0
        relative_pos = get_relative_transform(ego_matrix, ego_matrix)

        results = []

        # add ego-vehicle to results list
        # the format is category, extent*3, position*3, yaw, points_in_bbox, distance, id
        # the position is in lidar coordinates
        result = {"class": "Car",
                  "extent": [ego_dx[2], ego_dx[0], ego_dx[1]],  # TODO:
                  "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                  "yaw": relative_yaw,
                  "num_points": -1,
                  "distance": -1,
                  "speed": ego_speed,
                  "brake": ego_brake,
                  "id": int(ego_veh.id),
                  }
        results.append(result)

        # -----------------------------------------------------------
        # Other vehicles
        # -----------------------------------------------------------

        for vehicle in ego_nearby_vehicles:
            vehicle_transform = CarlaDataProvider.get_transform_after_tick(vehicle)
            vehicle_rotation = vehicle_transform.rotation

            vehicle_matrix = np.array(vehicle_transform.get_matrix())

            vehicle_extent = vehicle.bounding_box.extent
            dx = np.array([vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]) * 2.
            yaw = vehicle_rotation.yaw / 180 * np.pi

            relative_yaw = normalize_angle(yaw - ego_yaw)
            relative_pos = get_relative_transform(ego_matrix, vehicle_matrix)

            vehicle_control = vehicle.get_control()
            vehicle_velocity = vehicle.get_velocity()
            vehicle_speed = get_forward_speed(transform=vehicle_transform, velocity=vehicle_velocity)  # In m/s
            vehicle_brake = vehicle_control.brake

            # # filter bbox that didn't contain points of contains fewer points
            # if not lidar is None:
            #     num_in_bbox_points = self.get_points_in_bbox(ego_matrix, vehicle_matrix, dx, lidar)
            #     # print("num points in bbox", num_in_bbox_points)
            # else:
            #     num_in_bbox_points = -1
            num_in_bbox_points = -1

            distance = np.linalg.norm(relative_pos)

            result = {
                "class": "Car",
                "extent": [dx[2], dx[0], dx[1]],  # TODO
                "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                "yaw": relative_yaw,
                "num_points": int(num_in_bbox_points),
                "distance": distance,
                "speed": vehicle_speed,
                "brake": vehicle_brake,
                "id": int(vehicle.id),
            }
            results.append(result)

        # -----------------------------------------------------------
        # Route rdp
        # -----------------------------------------------------------
        max_len = 50
        waypoint_route = np.array([[node[0], node[1]] for node in waypoints])
        if len(waypoint_route) < max_len:
            max_len = len(waypoint_route)
        shortened_route = rdp(waypoint_route[:max_len], epsilon=0.5)

        # convert points to vectors
        vectors = shortened_route[1:] - shortened_route[:-1]
        midpoints = shortened_route[:-1] + vectors / 2.
        norms = np.linalg.norm(vectors, axis=1)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])

        for i, midpoint in enumerate(midpoints):
            # find distance to center of waypoint
            center_bounding_box = carla.Location(midpoint[0], midpoint[1], 0.0)
            transform = carla.Transform(center_bounding_box)
            route_matrix = np.array(transform.get_matrix())
            relative_pos = get_relative_transform(ego_matrix, route_matrix)
            distance = np.linalg.norm(relative_pos)

            # find distance to beginning of bounding box
            starting_bounding_box = carla.Location(shortened_route[i][0], shortened_route[i][1], 0.0)
            st_transform = carla.Transform(starting_bounding_box)
            st_route_matrix = np.array(st_transform.get_matrix())
            st_relative_pos = get_relative_transform(ego_matrix, st_route_matrix)
            st_distance = np.linalg.norm(st_relative_pos)

            # only store route boxes that are near the ego vehicle
            if i > 0 and st_distance > 30:
                continue

            length_bounding_box = carla.Vector3D(norms[i] / 2., ego_extent.y, ego_extent.z)
            bounding_box = carla.BoundingBox(transform.location, length_bounding_box)
            bounding_box.rotation = carla.Rotation(pitch=0.0,
                                                   yaw=angles[i] * 180 / np.pi,
                                                   roll=0.0)

            route_extent = bounding_box.extent
            dx = np.array([route_extent.x, route_extent.y, route_extent.z]) * 2.
            relative_yaw = normalize_angle(angles[i] - ego_yaw)

            # visualize subsampled route
            # self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1,
            #                             color=carla.Color(0, 255, 255, 255), life_time=(10.0/self.frame_rate_sim))

            result = {
                "class": "Route",
                "extent": [dx[2], dx[0], dx[1]],  # TODO
                "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                "yaw": relative_yaw,
                "centre_distance": distance,
                "starting_distance": st_distance,
                "id": i,
            }
            results.append(result)

        return results

    def get_input_batch(self, label_raw, target_point, traffic_light_hazard):
        sample = {'input': [], 'output': [], 'brake': [], 'waypoints': [], 'target_point': [], 'light': []}

        data = label_raw[1:]  # remove first element (ego vehicle)

        data_car = [[
            1.,  # type indicator for cars
            float(x['position'][0]) - float(label_raw[0]['position'][0]),
            float(x['position'][1]) - float(label_raw[0]['position'][1]),
            float(x['yaw'] * 180 / 3.14159265359),  # in degrees
            float(x['speed'] * 3.6),  # in km/h
            float(x['extent'][2]),
            float(x['extent'][1]),
        ] for x in data if x['class'] == 'Car']
        # if we use the far_node as target waypoint we need the route as input
        data_route = [
            [
                2.,  # type indicator for route
                float(x['position'][0]) - float(label_raw[0]['position'][0]),
                float(x['position'][1]) - float(label_raw[0]['position'][1]),
                float(x['yaw'] * 180 / 3.14159265359),  # in degrees
                float(x['id']),
                float(x['extent'][2]),
                float(x['extent'][1]),
            ]
            for j, x in enumerate(data)
            if x['class'] == 'Route'
               and float(x['id']) < 2]  # self.config['training']['max_NextRouteBBs']

        # we split route segment slonger than 10m into multiple segments
        # improves generalization
        data_route_split = []
        for route in data_route:
            if route[6] > 10:
                routes = split_large_BB(route, len(data_route_split))
                data_route_split.extend(routes)
            else:
                data_route_split.append(route)

        data_route = data_route_split[:2]  # self.config['training']['max_NextRouteBBs']

        assert len(data_route) <= 2, 'Too many routes'    # self.config['training']['max_NextRouteBBs']

        features = data_car + data_route

        sample['input'] = features

        # dummy data
        sample['output'] = features
        sample['light'] = traffic_light_hazard

        local_command_point = np.array([target_point[0], target_point[1]])
        sample['target_point'] = local_command_point

        batch = [sample]

        input_batch = generate_batch(batch)

        return input_batch
