#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：__init__.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
'''

# for planning scenario
from safebench.agent.dummy import DummyAgent
from safebench.agent.rl.sac import SAC
from safebench.agent.rl.ddpg import DDPG
from safebench.agent.rl.ppo import PPO
from safebench.agent.rl.td3 import TD3
from safebench.agent.basic import CarlaBasicAgent
from safebench.agent.behavior import CarlaBehaviorAgent

# for perception scenario
from safebench.agent.object_detection.yolov5 import YoloAgent
from safebench.agent.object_detection.faster_rcnn import FasterRCNNAgent

AGENT_POLICY_LIST = {
    'dummy': DummyAgent,
    'basic': CarlaBasicAgent,
    'behavior': CarlaBehaviorAgent,
    'yolo': YoloAgent,
    'sac': SAC,
    'ddpg': DDPG,
    'ppo': PPO,
    'td3': TD3,
    'faster_rcnn': FasterRCNNAgent,
}
