#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：dummy_policy.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
'''

from safebench.scenario.scenario_policy.base_policy import BasePolicy


class DummyPolicy(BasePolicy):
    name = 'dummy'
    type = 'unlearnable'

    """ This agent is used for scenarios that do not have controllable agents. """
    def __init__(self, config, logger):
        self.logger = logger
        self.logger.log('>> This scenario does not require policy model, using a dummy one', color='yellow')
        self.num_scenario = config['num_scenario']

    def train(self, replay_buffer):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, state, infos, deterministic=False):
        return [None] * self.num_scenario

    def get_init_action(self, scenario_config, deterministic=False):
        return [None] * self.num_scenario, None

    def load_model(self, scenario_configs=None):
        return None
