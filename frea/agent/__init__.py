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
from frea.agent.rl.ppo import PPO
from frea.agent.behavior import CarlaBehaviorAgent
from frea.agent.expert_disturb import CarlaExpertDisturbAgent
from frea.agent.expert.expert import CarlaExpertAgent
from frea.agent.PlanT.PlanT import PlanT


AGENT_POLICY_LIST = {
    'behavior': CarlaBehaviorAgent,
    'ppo': PPO,
    'expert': CarlaExpertAgent,
    'plant': PlanT,
    'expert_disturb': CarlaExpertDisturbAgent
}
