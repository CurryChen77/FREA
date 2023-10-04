#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：__init__.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
'''

from gym.envs.registration import register

register(
    id='carla-v0',
    entry_point='safebench.gym_carla.envs:CarlaEnv',
)
