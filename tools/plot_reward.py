#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：plot_reward.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


with open('../log/exp/exp_sac_standard_HJ-Reachability_seed_0/training_results/results.pkl', 'rb') as f:
    data = pkl.load(f)

episode = data['episode']
reward = data['episode_reward']

plt.plot(episode, reward)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.grid()
plt.xlim([0, 100])
plt.tight_layout()
plt.savefig('reward.png', dpi=300)
