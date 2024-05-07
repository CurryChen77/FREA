#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：base_policy.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""

class BasePolicy:
    name = 'base'
    type = 'unlearnable'

    """ This is the template for implementing the policy for a scenario. """
    def __init__(self, config, logger):
        self.continue_episode = 0
        self.num_scenario = config['num_scenario']

    def train(self, buffer, writer, e_i):
        raise NotImplementedError()

    def set_mode(self, mode):
        raise NotImplementedError()

    def get_action(self, state, infos, deterministic):
        raise NotImplementedError()
    
    def get_init_action(self, scenario_config, deterministic=False):
        raise NotImplementedError()

    def load_model(self, map_name, scenario_configs=None):
        raise NotImplementedError()

    def save_model(self, episode, map_name, buffer):
        raise NotImplementedError()