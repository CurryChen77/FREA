#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：test_finding_lanemarking.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/11/13
"""

import carla
import random
import carla
import time
import numpy as np


def test_map(center_vehicle, world, spectator, Map):

    center_transform = center_vehicle.get_transform()
    center_location = center_transform.location
    ego_waypoint = Map.get_waypoint(center_location)
    center_lane_waypoint = Map.get_waypoint(center_location, lane_type=carla.LaneType.NONE)
    print("center lane waypoint:", center_lane_waypoint)
    if ego_waypoint:
        world.debug.draw_point(ego_waypoint.transform.location + carla.Location(z=4), size=0.1, life_time=0.11)
        if center_lane_waypoint:
            world.debug.draw_point(center_lane_waypoint.transform.location + carla.Location(z=4), size=0.1, life_time=0.11, color=carla.Color(0, 255,0,0))
            left_lane_marking = center_lane_waypoint.left_lane_marking.type
            print('left lane marking type', left_lane_marking)
            right_lane_marking = center_lane_waypoint.right_lane_marking.type
            print('right lane marking type', right_lane_marking)
    spectator.set_transform(carla.Transform(
        center_location + carla.Location(x=-3, z=40), carla.Rotation(yaw=center_transform.rotation.yaw,pitch=-80.0)
    ))


def main():
    actor_list = []
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        Map = world.get_map()
        blueprint_library = world.get_blueprint_library()
        bp1 = random.choice(blueprint_library.filter('vehicle'))
        if bp1.has_attribute('color'):
            color1 = random.choice(bp1.get_attribute('color').recommended_values)
            bp1.set_attribute('color', color1)
        transform1 = random.choice(world.get_map().get_spawn_points())
        bp1.set_attribute('role_name', 'hero')
        bp1.set_attribute('role_name', 'background')
        ego = world.spawn_actor(bp1, transform1)

        actor_list.append(ego)
        print('created ego %s' % ego.type_id)
        # Let's put the vehicle to drive around.
        ego.set_autopilot(True)
        spectator = world.get_spectator()


        for _ in range(250000):
            world.tick()
            test_map(ego, world, spectator, Map)  # Test the function of generating box2origin matrix(4x4)


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