#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：__init__.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""

# collect policy models from scenarios
from frea.scenario.scenario_policy.dummy_policy import DummyPolicy
from frea.scenario.scenario_policy.rl.ppo import PPO
from frea.scenario.scenario_policy.rl.fppo_adv import FPPOAdv
from frea.scenario.scenario_policy.rl.fppo_rs import FPPORs


SCENARIO_POLICY_LIST = {
    'standard': DummyPolicy,
    'scenic': DummyPolicy,
    'ppo': PPO,
    'fppo_adv': FPPOAdv,
    'fppo_rs': FPPORs,
}
