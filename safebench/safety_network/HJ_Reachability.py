#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：HJ-Reachability
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/9
"""

import os

import numpy as np
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
        x = torch.cat((x, a), -1)  # combination x and a
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class HJR:
    name = 'HJR'
    type = 'onpolicy'

    def __init__(self, config, logger):
        self.logger = logger
        self.min_dis_threshold = config['min_dis_threshold']
        self.buffer_start_training = config['buffer_start_training']
        self.lr = config['lr']
        self.obs_type = config['obs_type']
        self.continue_episode = 0
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.acc_range = config['acc_range']
        self.steer_range = config['steer_range']

        self.batch_size = config['batch_size']
        self.train_iteration = config['train_iteration']
        self.agent_info = 'ego_' + config['agent_policy_type'] + "_" + config['agent_obs_type']

        self.agent_policy_type = config['agent_policy_type']
        self.scenario_policy_type = config['scenario_policy_type']
        model_name = "EgoPolicy_" + self.agent_policy_type+"_" + config['agent_obs_type'] + "_CBVPolicy_" + self.scenario_policy_type + "_" + "HJR"
        self.model_path = os.path.join(config['ROOT_DIR'], config['model_path'], model_name)
        self.scenario_id = config['scenario_id']

        self.Qh_net = CUDA(Q(self.state_dim, self.action_dim))  # the Q network of constrain
        self.Qh_optimizer = optim.Adam(self.Qh_net.parameters(), lr=self.lr)  # the corresponding optimizer of Qh
        self.Qh_criterion = nn.MSELoss()  # the corresponding optimizer of Qh

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.Qh_net.train()
        elif mode == 'eval':
            self.Qh_net.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')

    def find_min_Qh(self, n_state):
        def Qh(acceleration, steering, next_state):
            acceleration = torch.from_numpy(acceleration)
            steering = torch.from_numpy(steering)
            safest_action = CUDA(torch.cat((acceleration, steering), dim=1).type(next_state.dtype))
            Qh_value = self.Qh_net(next_state, safest_action)
            # the object is to min the Qh value of each state (equal to min sum of overall Qh)
            return Qh_value.sum().item()

        parametrization = ng.p.Instrumentation(
            acceleration=ng.p.Array(shape=(self.batch_size, 1)).set_bounds(-3.0, 3.0),
            steering=ng.p.Array(shape=(self.batch_size, 1)).set_bounds(-0.3, 0.3),
            next_state=n_state
        )

        optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=100)
        recommendation = optimizer.minimize(Qh)

        acc = recommendation.kwargs['acceleration']
        steer = recommendation.kwargs['steering']
        acc = torch.from_numpy(acc)
        steer = torch.from_numpy(steer)
        best_action = CUDA(torch.cat((acc, steer), dim=1).type(n_state.dtype))
        min_Qs = self.Qh_net(n_state, best_action)
        return min_Qs

    def train(self, replay_buffer, writer, e_i):

        for _ in range(self.train_iteration):
            # sample replay buffer
            batch = replay_buffer.sample(self.batch_size)
            if self.obs_type == 'plant':
                bn_s_ = CUDA(torch.FloatTensor(batch['n_encoded_state'])).reshape(self.batch_size, -1)  # next state are the next encoded state
            elif self.obs_type == 'actor_info':
                bn_s_ = CUDA(torch.FloatTensor(batch['n_actor_info'])).reshape(self.batch_size, -1)  # the next state are the next actor infos
            else:
                bn_s_ = CUDA(torch.FloatTensor(batch['n_state']))  # using the next state for agent

            bn_d = CUDA(torch.FloatTensor(1-batch['done'])).unsqueeze(-1)  # [B, 1]
            # the ego min distance from the infos
            bn_min_dis = CUDA(torch.FloatTensor(batch['ego_min_dis'])).unsqueeze(-1)

            # h = threshold - min_dis, if h > 0 unsafe, else safe
            bn_h = torch.zeros_like(bn_min_dis).fill_(self.min_dis_threshold) - bn_min_dis

            # the Qh calculation from RCRL
            excepted_Qh = self.find_min_Qh(bn_s_)  # find the min Qh of the next state

            Qh_target_terminal = bn_h
            Qh_target_non_terminal = torch.maximum(bn_h, excepted_Qh)  # from RCRL
            Qh_target = torch.where(bn_d.bool(), Qh_target_terminal, Qh_target_non_terminal)

            # the Qh loss
            Qh_loss = self.Qh_criterion(excepted_Qh, Qh_target.detach())  # J_Qh
            Qh_loss = Qh_loss.mean()
            writer.add_scalar("HJR_Qh_loss", Qh_loss, e_i)

            self.Qh_optimizer.zero_grad()
            Qh_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.Qh_net.parameters(), 0.5)
            self.Qh_optimizer.step()

        # reset buffer
        replay_buffer.reset_buffer()

    def save_model(self, episode, map_name):
        states = {
            'Qh_net': self.Qh_net.state_dict()
        }
        scenario_name = "all" if self.scenario_id is None else 'scenario' + str(self.scenario_id)
        save_dir = os.path.join(self.model_path, scenario_name+"_"+map_name)
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f'model.HJR.{episode:04}.torch')
        self.logger.log(f'>> Saving {self.name} model to {filepath}')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)

    def load_model(self, map_name, episode=None):
        scenario_name = "all" if self.scenario_id is None else 'scenario' + str(self.scenario_id)
        load_dir = os.path.join(self.model_path, scenario_name+"_"+map_name)
        if episode is None:
            episode = -1
            for _, _, files in os.walk(load_dir):
                for name in files:
                    if fnmatch(name, "*torch"):
                        cur_episode = int(name.split(".")[-2])
                        if cur_episode > episode:
                            episode = cur_episode
        filepath = os.path.join(load_dir, f'model.HJR.{episode:04}.torch')
        if os.path.isfile(filepath):
            self.logger.log(f'>> Loading Safety network {self.name} from {os.path.basename(filepath)}')
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.Qh_net.load_state_dict(checkpoint['Qh_net'])
            self.continue_episode = episode
        else:
            self.logger.log(f'>> No {self.name} model found at {filepath}', 'red')
            self.continue_episode = 0