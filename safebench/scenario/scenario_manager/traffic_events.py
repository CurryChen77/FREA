#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：traffic_events.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""

from enum import Enum


class TrafficEventType(Enum):
    """
        This enum represents different traffic events that occur during driving.
    """
    NORMAL_DRIVING = 0
    COLLISION_STATIC = 1
    COLLISION_VEHICLE = 2
    COLLISION_PEDESTRIAN = 3
    ROUTE_DEVIATION = 4
    ROUTE_COMPLETION = 5
    ROUTE_COMPLETED = 6
    TRAFFIC_LIGHT_INFRACTION = 7
    WRONG_WAY_INFRACTION = 8
    ON_SIDEWALK_INFRACTION = 9
    STOP_INFRACTION = 10
    OUTSIDE_LANE_INFRACTION = 11
    OUTSIDE_ROUTE_LANES_INFRACTION = 12
    VEHICLE_BLOCKED = 13


class TrafficEvent(object):
    def __init__(self, event_type, message=None, dictionary=None):
        """
            Initialize object
                :param event_type: TrafficEventType defining the type of traffic event
                :param message: optional message to inform users of the event
                :param dictionary: optional dictionary with arbitrary keys and values
        """
        self._type = event_type
        self._message = message
        self._dict = dictionary

    def get_type(self):
        return self._type

    def get_message(self):
        if self._message:
            return self._message
        return ""

    def set_message(self, message):
        self._message = message

    def get_dict(self):
        return self._dict

    def set_dict(self, dictionary):
        self._dict = dictionary
