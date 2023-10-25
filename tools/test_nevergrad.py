#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：test_nevergrad.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/10
"""

import nevergrad as ng
import torch
import torch.nn as nn
import torch.optim as optim
from fnmatch import fnmatch
from torch.distributions import Normal
import nevergrad as ng
from safebench.util.torch_util import CUDA, CPU, kaiming_init
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Q(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.fc1 = nn.Linear(state_dim+action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.apply(kaiming_init)

    def forward(self, x, a):
        x = x.reshape(-1, self.state_dim)
        a = a.reshape(-1, self.action_dim)
        x = torch.cat((x, a), -1) # combination x and a
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # Instrumentation class is used for functions with multiple inputs
    # (positional and/or keywords)
    Qh = Q(state_dim=8, action_dim=2)
    state = torch.randn(64, 8)

    def fake_training(acc, steer, next_state):
        # optimal for learning_rate=0.2, batch_size=4, architecture="conv"
        acc = torch.from_numpy(acc)
        steer = torch.from_numpy(steer)
        action = torch.cat((acc, steer), dim=1).type(torch.float32)

        q = Qh(next_state, action)
        return q.sum().item()

    parametrization = ng.p.Instrumentation(

        acc=ng.p.Array(shape=(64, 1)).set_bounds(-3.0, 3.0),

        steer=ng.p.Array(shape=(64, 1)).set_bounds(-0.3, 0.3),
        next_state=state
    )

    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=100)
    recommendation = optimizer.minimize(fake_training)

