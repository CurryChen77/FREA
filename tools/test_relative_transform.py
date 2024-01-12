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


def get_relative_info(actor, ego):
    """
        Transform vehicle to ego coordinate
        :param global_info: surrounding vehicle's global info
        :param ego_info: ego vehicle info
        :return: tuple of the pose of the surrounding vehicle in ego coordinate
    """
    actor_transform = actor.get_transform()
    actor_velocity = actor.get_velocity()
    x = actor_transform.location.x
    y = actor_transform.location.y
    yaw = actor_transform.rotation.yaw / 180 * np.pi
    vx = actor_velocity.x
    vy = actor_velocity.y

    ego_transform = ego.get_transform()
    ego_velocity = ego.get_velocity()
    ego_x = ego_transform.location.x
    ego_y = ego_transform.location.y
    ego_yaw = ego_transform.rotation.yaw / 180 * np.pi
    ego_vx = ego_velocity.x
    ego_vy = ego_velocity.y

    R = np.array([[np.cos(ego_yaw), np.sin(ego_yaw)],
                                [-np.sin(ego_yaw), np.cos(ego_yaw)]])
    location_local = R.dot(np.array([x - ego_x, y - ego_y]))
    velocity_local = np.linalg.norm(np.array([vx - ego_vx, vy - ego_vy]))
    yaw_local = normalize_angle(yaw - ego_yaw)
    local_info = [location_local[0], location_local[1], yaw_local, velocity_local]
    print("carla API local info", local_info)


def get_relative_transform(ego_matrix, vehicle_matrix):
    """
    return the relative transform from ego_pose to vehicle pose
    """
    relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
    rot = ego_matrix[:3, :3].T
    relative_pos = rot @ relative_pos

    # transform to right-handed system
    relative_pos[1] = - relative_pos[1]
    return relative_pos


def normalize_angle(x):
    x = x % (2 * np.pi)  # force in range [0, 2 pi)
    if x > np.pi:  # move to [-pi, pi)
        x -= 2 * np.pi
    return x


def get_relative_info_use_carla(actor, ego):
    ego_transform = ego.get_transform()
    ego_velocity = ego.get_velocity()
    ego_matrix = np.array(ego.get_transform().get_matrix())

    actor_velocity = actor.get_velocity()
    actor_transform = actor.get_transform()
    actor_matrix = np.array(actor.get_transform().get_matrix())

    relative_pos = get_relative_transform(ego_matrix, actor_matrix)

    yaw = actor_transform.rotation.yaw / 180 * np.pi
    ego_yaw = ego_transform.rotation.yaw / 180 * np.pi
    yaw_local = normalize_angle(yaw - ego_yaw)

    rel_vel = ego_velocity.distance_2d(actor_velocity)

    local_info = [relative_pos[0], relative_pos[1], yaw_local, rel_vel]
    print("local info", local_info)





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
            print("----------------------------------------------------------------")
            get_relative_info_use_carla(bv, ego)
            get_relative_info(bv, ego)
            print("----------------------------------------------------------------")

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

