#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：__init__.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""

# for planning scenario
from safebench.agent.rl.sac import SAC
from safebench.agent.rl.rc_sac import RCSAC
from safebench.agent.rl.ddpg import DDPG
from safebench.agent.rl.ppo import PPO
from safebench.agent.rl.td3 import TD3
from safebench.agent.behavior import CarlaBehaviorAgent
from safebench.agent.expert.expert import CarlaExpertAgent
from safebench.agent.PlanT.PlanT import PlanT


AGENT_POLICY_LIST = {
    'behavior': CarlaBehaviorAgent,
    'sac': SAC,
    'rc_sac': RCSAC,
    'ddpg': DDPG,
    'ppo': PPO,
    'td3': TD3,
    'expert': CarlaExpertAgent,
    'plant': PlanT
}
