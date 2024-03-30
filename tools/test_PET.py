#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：test_PET
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/3/29
"""
import carla
import random
import time
import numpy as np


def get_color(attention):
    colors = [
        (255, 255, 255, 255),
        # (220, 228, 180, 255),
        # (190, 225, 150, 255),
        (240, 240, 210, 255),
        # (190, 219, 96, 255),
        (240, 220, 150, 255),
        # (170, 213, 79, 255),
        (240, 210, 110, 255),
        # (155, 206, 62, 255),
        (240, 200, 70, 255),
        # (162, 199, 44, 255),
        (240, 190, 30, 255),
        # (170, 192, 20, 255),
        (240, 185, 0, 255),
        # (177, 185, 0, 255),
        (240, 181, 0, 255),
        # (184, 177, 0, 255),
        (240, 173, 0, 255),
        # (191, 169, 0, 255),
        (240, 165, 0, 255),
        # (198, 160, 0, 255),
        (240, 156, 0, 255),
        # (205, 151, 0, 255),
        (240, 147, 0, 255),
        # (212, 142, 0, 255),
        (240, 137, 0, 255),
        # (218, 131, 0, 255),
        (240, 126, 0, 255),
        # (224, 120, 0, 255),
        (240, 114, 0, 255),
        # (230, 108, 0, 255),
        (240, 102, 0, 255),
        # (235, 95, 0, 255),
        (240, 88, 0, 255),
        # (240, 80, 0, 255),
        (242, 71, 0, 255),
        # (244, 61, 0, 255),
        (246, 49, 0, 255),
        # (247, 34, 0, 255),
        (248, 15, 0, 255),
        (249, 6, 6, 255),
    ]

    ix = int(attention * (len(colors) - 1))
    return colors[ix]


def check_ray_intersection(vector_1, vector_2, loc_1, loc_2):
    relative_vector21 = loc_2 - loc_1
    relative_vector12 = loc_1 - loc_2
    if vector_1.dot(relative_vector21) > 0 and vector_2.dot(relative_vector12) > 0:
        return True
    else:
        return False


def line_intersection(vector_1, vector_2, loc_1, loc_2):
    # carla to numpy
    # Convert to numpy arrays
    P1 = np.array([loc_1.x, loc_1.y])
    V1 = np.array([vector_1.x, vector_1.y])
    P2 = np.array([loc_2.x, loc_2.y])
    V2 = np.array([vector_2.x, vector_2.y])

    # Set up the linear equations
    A = np.array([V1, -V2]).T
    b = P2 - P1

    # Attempt to solve the linear system A * t = b for t
    try:
        t = np.linalg.solve(A, b)
        # t[0] corresponds to the parameter for the first ray,
        # and t[1] corresponds to the parameter for the second ray.
        # Check if both parameters are positive, indicating an intersection in the direction of both rays.
        if t[0] >= 0 and t[1] >= 0:
            intersection_point = P1 + t[0] * V1
            return tuple(intersection_point)
        else:
            return None
    except np.linalg.LinAlgError:
        # The system is linearly dependent, indicating parallel or coincident lines.
        return None


def get_forward_speed(transform, velocity):
    """
        Convert the vehicle transform directly to forward speed
    """

    vel_np = np.array([velocity.x, velocity.y, velocity.z])
    pitch = np.deg2rad(transform.rotation.pitch)
    yaw = np.deg2rad(transform.rotation.yaw)
    orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    speed = np.dot(vel_np, orientation)
    return speed


def test_PET(ego, bv, world):
    ego_extent = ego.bounding_box.extent
    ego_trans = ego.get_transform()
    bv_trans = bv.get_transform()
    ego_loc = ego_trans.location
    bv_loc = bv_trans.location
    ego_forward_vec = ego_trans.get_forward_vector()
    ego_vel = get_forward_speed(transform=ego_trans, velocity=ego.get_velocity())  # In m/s
    bv_vel = get_forward_speed(transform=bv_trans, velocity=bv.get_velocity())  # In m/s

    ego_yaw = ego_trans.rotation.yaw / 180 * np.pi
    bv_yaw = bv_trans.rotation.yaw / 180 * np.pi
    bv_forward_vec = bv_trans.get_forward_vector()
    if check_ray_intersection(ego_forward_vec, bv_forward_vec, ego_loc, bv_loc):
        intersection_point = line_intersection(ego_forward_vec, bv_forward_vec, ego_loc, bv_loc)
        if intersection_point is not None:
            world.debug.draw_point(carla.Location(x=intersection_point[0], y=intersection_point[1], z=4), size=0.1, color=carla.Color(0, 255, 0, 0), life_time=0.01)

            ego_intersection_dis = ego_loc.distance(carla.Location(x=intersection_point[0], y=intersection_point[1], z=ego_loc.z))
            bv_intersection_dis = bv_loc.distance(carla.Location(x=intersection_point[0], y=intersection_point[1], z=bv_loc.z))

            ego_time = ego_intersection_dis / ego_vel
            bv_time = bv_intersection_dis / bv_vel

            delta_yaw = abs(ego_yaw - bv_yaw)
            delta_yaw = 2 * np.pi - delta_yaw if delta_yaw > np.pi else delta_yaw

            delta_x = ego_extent.y * (1 / abs(np.tan(delta_yaw)) + 1 / np.sin(delta_yaw)) + ego_extent.x
            if ego_time >= bv_time:
                t_large, t_small, v_large, v_small = ego_time, bv_time, ego_vel, bv_vel
            else:
                t_large, t_small, v_large, v_small = bv_time, ego_time, bv_vel, ego_vel
            if v_large > 0 and v_small > 0:
                t_large_prime = t_large - delta_x / v_large
                t_small_prime = t_small + delta_x / v_small
                PET = round(t_large_prime - t_small_prime, 2)
                print("PET:", PET)

def main():
    actor_list = []
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        bp1 = random.choice(blueprint_library.filter('vehicle'))
        bp2 = random.choice(blueprint_library.filter('vehicle'))

        spectator = world.get_spectator()

        if bp1.has_attribute('color'):
            color1 = random.choice(bp1.get_attribute('color').recommended_values)
            bp1.set_attribute('color', color1)
        if bp2.has_attribute('color'):
            color2 = random.choice(bp2.get_attribute('color').recommended_values)
            bp2.set_attribute('color', color2)
        transform_list = world.get_map().get_spawn_points()
        transform1 = transform_list[65]
        transform2 = transform_list[276]
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

        for _ in range(4500):
            world.tick()
            ego_transform = ego.get_transform()
            spectator.set_transform(carla.Transform(
                    ego_transform.location + carla.Location(x=-3, z=50), carla.Rotation(yaw=ego_transform.rotation.yaw, pitch=-80.0)
                ))
            test_PET(ego, bv, world)

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