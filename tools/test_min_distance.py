#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：test_min_distance.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/12
"""


import carla
import random
import time
import numpy as np
from distance3d import gjk, geometry, colliders


def compute_R(rotation):
    pitch_rad = np.radians(rotation.pitch)
    yaw_rad = np.radians(rotation.yaw)
    roll_rad = np.radians(rotation.roll)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll_rad), -np.sin(roll_rad)],
                   [0, np.sin(roll_rad), np.cos(roll_rad)]])
    Ry = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                   [0, 1, 0],
                   [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
    Rz = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                   [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                   [0, 0, 1]])

    rotation_matrix = Rx @ Ry @ Rz
    return rotation_matrix


def compute_box2origin(vehicle_box, vehicle_transform):
    vehicle_location = vehicle_transform.location
    bbox_location = vehicle_box.location
    # the transition vector should be the sum of vehicle_location(global) and bbox_location(local)
    t = np.array([
        bbox_location.x + vehicle_location.x,
        bbox_location.y + vehicle_location.y,
        bbox_location.z + vehicle_location.z])
    r = compute_R(vehicle_transform.rotation)  # the rotation matrix
    extent = vehicle_box.extent
    size = np.array([extent.x * 2, extent.y * 2, extent.z * 2])

    box2origin = np.zeros((4, 4))
    box2origin[:3, :3] = r
    box2origin[:3, 3] = t
    box2origin[3, 3] = 1.0
    return box2origin, size


def test_box2origin(vehicle):
    vehicle_bbox = vehicle.bounding_box
    print("-----------------------------------------------------")
    print("Vehicle bbox location (local):", vehicle_bbox.location)
    vehicle_transform = vehicle.get_transform()
    print("Vehicle location (global):", vehicle_transform.location)
    carla_ver = vehicle_bbox.get_world_vertices(vehicle_transform)
    carla_vertices = np.zeros((8, 3))
    for i in range(len(carla_ver)):
        carla_vertices[i, 0] = round(carla_ver[i].x, 3)
        carla_vertices[i, 1] = round(carla_ver[i].y, 3)
        carla_vertices[i, 2] = round(carla_ver[i].z, 3)
    box2origin, size = compute_box2origin(vehicle_bbox, vehicle_transform)
    print('box2origin \n', box2origin)
    print('size', size)
    vertices = geometry.convert_box_to_vertices(box2origin, size)
    print("True vertices from carla API: \n", carla_vertices)
    print('Calculated vertices from box2origin\n', vertices)
    print("-----------------------------------------------------")


def test_min_distance(ego, bv):
    ego_bbox = ego.bounding_box
    bv_bbox = bv.bounding_box
    ego_transform = ego.get_transform()
    bv_transform = bv.get_transform()
    start = time.time()
    box2origin_ego, size_ego = compute_box2origin(ego_bbox, ego_transform)
    box2origin_bv, size_bv = compute_box2origin(bv_bbox, bv_transform)
    # min distance
    box_collider_ego = colliders.Box(box2origin_ego, size_ego)
    box_collider_bv = colliders.Box(box2origin_bv, size_bv)
    dist, closest_point_box, closest_point_box2, _ = gjk.gjk(
        box_collider_ego, box_collider_bv)
    end = time.time()
    cal_min_dis_time = end - start
    print("-----------------------------------------------------")
    print("min distance using GJK method: ", dist)

    # center distance
    ego_location = ego_transform.location
    start1 = time.time()
    distance = ego_location.distance(bv_transform.location)
    end1 = time.time()
    cal_distance_time = end1 - start1
    print("center point distance using carla API: ", distance)

    # corresponding time
    print("min distance time: ", cal_min_dis_time)
    print("center point distance time: ", cal_distance_time)
    print("-----------------------------------------------------")


def main():
    actor_list = []
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        bp1 = random.choice(blueprint_library.filter('vehicle'))
        bp2 = random.choice(blueprint_library.filter('vehicle'))
        if bp1.has_attribute('color'):
            color1 = random.choice(bp1.get_attribute('color').recommended_values)
            bp1.set_attribute('color', color1)
        if bp2.has_attribute('color'):
            color2 = random.choice(bp2.get_attribute('color').recommended_values)
            bp2.set_attribute('color', color2)
        transform1 = random.choice(world.get_map().get_spawn_points())
        transform2 = random.choice(world.get_map().get_spawn_points())
        bp1.set_attribute('role_name', 'hero')
        bp1.set_attribute('role_name', 'background')
        ego = world.spawn_actor(bp1, transform1)
        bv = world.spawn_actor(bp2, transform2)
        actor_list.append(ego)
        actor_list.append(bv)
        print('created ego %s' % ego.type_id)
        print('created bv %s' % bv.type_id)
        # Let's put the vehicle to drive around.
        ego.set_autopilot(True)
        bv.set_autopilot(True)

        for _ in range(2500):
            world.tick()
            # test_box2origin(ego)  # Test the function of generating box2origin matrix(4x4)
            test_min_distance(ego, bv)

    finally:
        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')

