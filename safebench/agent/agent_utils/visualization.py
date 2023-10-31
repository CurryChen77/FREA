#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：visualization.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/30
"""

from rdp import rdp
import carla
import numpy as np


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


def draw_route(world, vehicle=None, waypoint_route=None, bounding_box=None):
    if bounding_box is None:
        ego_transform = vehicle.get_transform()
        ego_matrix = np.array(ego_transform.get_matrix())
        ego_extent = vehicle.bounding_box.extent
        max_len = 50
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
            world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.15,
                                 color=carla.Color(0, 255, 255, 255), life_time=(0.11))
    else:
        # visualize subsampled route
        world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.15,
                                   color=carla.Color(0, 255, 255, 255), life_time=(0.11))