#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：fppo_adv.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/2/21
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""

from safebench.scenario.scenario_policy.rl.ppo import PPO


class FPPORs(PPO):
    name = 'fppo_rs'
    type = 'onpolicy'

    def __init__(self, config, logger):
        super(FPPORs, self).__init__(config, logger)


