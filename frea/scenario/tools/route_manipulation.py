#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：route_manipulation.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""

import math
import xml.etree.ElementTree as ET

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption


def _location_to_gps(lat_ref, lon_ref, location):
    """
        Convert from world coordinates to GPS coordinates
            :param lat_ref: latitude reference for the current map
            :param lon_ref: longitude reference for the current map
            :param location: location to translate
            :return: dictionary with lat, lon and height
    """

    EARTH_RADIUS_EQUA = 6378137.0   # pylint: disable=invalid-name
    scale = math.cos(lat_ref * math.pi / 180.0)
    mx = scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    my = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
    mx += location.x
    my -= location.y

    lon = mx * 180.0 / (math.pi * EARTH_RADIUS_EQUA * scale)
    lat = 360.0 * math.atan(math.exp(my / (EARTH_RADIUS_EQUA * scale))) / math.pi - 90.0
    z = location.z

    return {'lat': lat, 'lon': lon, 'z': z}


def location_route_to_gps(route, lat_ref, lon_ref):
    """
        Locate each waypoint of the route into gps, (lat long ) representations.
            :param route:
            :param lat_ref:
            :param lon_ref:
            :return:
    """
    gps_route = []

    for transform, connection in route:
        gps_point = _location_to_gps(lat_ref, lon_ref, transform.location)
        gps_route.append((gps_point, connection))

    return gps_route


def _get_latlon_ref(world):
    """
        Convert from waypoints world coordinates to CARLA GPS coordinates
            :return: tuple with lat and lon coordinates
    """
    xodr = world.get_map().to_opendrive()
    tree = ET.ElementTree(ET.fromstring(xodr))

    # default reference
    lat_ref = 42.0
    lon_ref = 2.0

    for opendrive in tree.iter("OpenDRIVE"):
        for header in opendrive.iter("header"):
            for georef in header.iter("geoReference"):
                if georef.text:
                    str_list = georef.text.split(' ')
                    for item in str_list:
                        if '+lat_0' in item:
                            lat_ref = float(item.split('=')[1])
                        if '+lon_0' in item:
                            lon_ref = float(item.split('=')[1])
    return lat_ref, lon_ref


def downsample_route(route, sample_factor):
    """
        Downsample the route by some factor.
            :param route: the trajectory , has to contain the waypoints and the road options
            :param sample_factor: Maximum distance between samples
            :return: returns the ids of the final route that can
    """

    ids_to_sample = []
    prev_option = None
    dist = 0

    for i, point in enumerate(route):
        curr_option = point[1]

        # Lane changing
        if curr_option in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            ids_to_sample.append(i)
            dist = 0

        # When road option changes
        elif prev_option != curr_option and prev_option not in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            ids_to_sample.append(i)
            dist = 0

        # After a certain max distance
        elif dist > sample_factor:
            ids_to_sample.append(i)
            dist = 0

        # At the end
        elif i == len(route) - 1:
            ids_to_sample.append(i)
            dist = 0

        # Compute the distance traveled
        else:
            curr_location = point[0].location
            prev_location = route[i - 1][0].location
            dist += curr_location.distance(prev_location)

        prev_option = curr_option

    return ids_to_sample


def interpolate_trajectory(world, waypoints_trajectory, hop_resolution=1.0):
    """
        Given some raw keypoints interpolate a full dense trajectory to be used by the user.
            :param world: an reference to the CARLA world so we can use the planner
            :param waypoints_trajectory: the current coarse trajectory
            :param hop_resolution: is the resolution, how dense is the provided trajectory going to be made
            :return: the full interpolated route both in GPS coordinates and also in its original form.
    """

    grp = GlobalRoutePlanner(world.get_map(), hop_resolution)
    route = []

    if len(waypoints_trajectory) == 1:
        route.append((waypoints_trajectory[0], RoadOption.VOID))

    for i in range(len(waypoints_trajectory) - 1):   # Goes until the one before the last.
        waypoint = waypoints_trajectory[i]
        waypoint_next = waypoints_trajectory[i + 1]
        interpolated_trace = grp.trace_route(waypoint, waypoint_next)
        for wp_tuple in interpolated_trace:
            route.append((wp_tuple[0].transform, wp_tuple[1]))

    lat_ref, lon_ref = _get_latlon_ref(world)

    return location_route_to_gps(route, lat_ref, lon_ref), route
