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
from safebench.scenario.scenario_policy.dummy_policy import DummyPolicy
from safebench.scenario.scenario_policy.rl.sac import SAC
from safebench.scenario.scenario_policy.rl.ppo import PPO
from safebench.scenario.scenario_policy.rl.fppo import FPPO
from safebench.scenario.scenario_policy.rl.fppo_lag import FPPOLag


SCENARIO_POLICY_LIST = {
    'standard': DummyPolicy,
    'scenic': DummyPolicy,
    'sac': SAC,
    'ppo': PPO,
    'fppo': FPPO,
    'fppo_lag': FPPOLag
}
