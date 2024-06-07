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
from frea.gym_carla.net import CriticPPO, CriticTwin
from frea.util.torch_util import CUDA, CPU


class HJR:
    name = 'HJR'
    type = 'onpolicy'

    def __init__(self, config, logger):
        self.logger = logger
        self.min_dis_threshold = config['min_dis_threshold']
        self.lr = config['lr']
        self.obs_type = config['obs_type']
        self.continue_episode = 0
        self.state_dim = config['feasibility_state_dim']
        self.action_dim = config['agent_action_dim']
        self.acc_range = config['acc_range']
        self.steer_range = config['steer_range']
        self.dims = config['dims']
        self.max_train_episode = config['train_episode']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.expectile = config['expectile']
        self.M = config['M']
        self.seed = config['seed']

        self.batch_size = config['batch_size']
        self.model_path = os.path.join(config['ROOT_DIR'], config['model_path'], 'min_dis_threshold_' + str(self.min_dis_threshold) + '_seed' + str(self.seed))
        self.scenario_id = config['scenario_id']

        self.Qh_net = CUDA(CriticTwin(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))  # the Q network of constraint
        self.Qh_target_net = CUDA(CriticTwin(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))  # the Q network of constraint
        self.Qh_optimizer = optim.Adam(self.Qh_net.parameters(), lr=self.lr, eps=1e-5)  # the corresponding optimizer of Qh
        self.Qh_criterion = nn.MSELoss()

        self.Vh_net = CUDA(CriticPPO(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))  # the V network of constraint
        self.Vh_optimizer = torch.optim.Adam(self.Vh_net.parameters(), lr=self.lr, eps=1e-5)  # the corresponding optimizer of Vh

        # copy parameters of the Qh network to Qh target network
        for target_param, param in zip(self.Qh_target_net.parameters(), self.Qh_net.parameters()):
            target_param.data.copy_(param.data)

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.Qh_net.train()
            self.Vh_net.train()
        elif mode == 'eval':
            self.Qh_net.eval()
            self.Vh_net.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')

    def lr_decay(self, e_i):
        lr_now = self.lr * (1 - e_i / self.max_train_episode)
        for p in self.Qh_optimizer.param_groups:
            p['lr'] = lr_now
        for p in self.Vh_optimizer.param_groups:
            p['lr'] = lr_now

    def get_feasibility_Vs(self, state):
        state = state.reshape(state.shape[0], -1)
        feasibility_value = self.Vh_net(state)
        return feasibility_value

    def get_feasibility_Qs(self, state, action):
        state = state.reshape(state.shape[0], -1)
        feasibility_Q = self.Qh_net(state, action)
        return feasibility_Q

    def get_feasibility_advantage(self, state, action):
        state = state.reshape(state.shape[0], -1)
        feasibility_value = self.Vh_net(state)
        feasibility_Q = self.Qh_net(state, action)
        feasibility_advantage = feasibility_Q - feasibility_value
        return feasibility_advantage

    @staticmethod
    def safe_expectile_loss(diff, expectile=0.8):
        weight = torch.where(diff < 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def compute_Vh_loss(self, state, action):
        # the Qh is about the constraint h, so lower means better, this is different from the reward, higher the better
        Qh_max = self.Qh_target_net.get_q_max(state, action)
        Vh = self.Vh_net(state)
        Vh_loss = self.safe_expectile_loss(diff=Qh_max - Vh, expectile=self.expectile).mean()
        return Vh_loss, Vh.mean()

    def compute_Qh_loss(self, h, state, action, next_state, undone):
        next_Vh = self.Vh_net(next_state)
        Qh_nonterminal = (1. - self.gamma) * h + self.gamma * torch.maximum(h, next_Vh)
        target_Qh = Qh_nonterminal * undone + h * (1. - undone)
        Qh1, Qh2 = self.Qh_net.get_q1_q2(state, action)
        Qh_loss = self.Qh_criterion(Qh1, target_Qh) + self.Qh_criterion(Qh2, target_Qh)
        return Qh_loss

    def train(self, buffer, writer, e_i):

        with torch.no_grad():
            # learning rate decay
            self.lr_decay(e_i)  # add the learning rate decay

            batch = buffer.sample(self.batch_size)

            state = batch['obs'].reshape((-1, self.state_dim))
            next_state = batch['next_obs'].reshape((-1, self.state_dim))
            action = batch['actions']
            undone = 1-batch['dones']
            # the ego min distance from the infos
            min_dis = batch['ego_min_dis']

            # h is -1.0 when Ego is safe, else, h is M
            h = torch.where(min_dis <= float(self.min_dis_threshold), self.M, -1)

            del min_dis

        # get the Vh loss
        Vh_loss, Vh_mean = self.compute_Vh_loss(state=state, action=action)
        writer.add_scalar("HJR_Vh_loss", Vh_loss, e_i)
        writer.add_scalar("HJR_Vh_mean", Vh_mean, e_i)
        # update the Vh net
        self.Vh_optimizer.zero_grad()
        Vh_loss.backward()
        nn.utils.clip_grad_norm_(self.Vh_net.parameters(), 0.5)
        self.Vh_optimizer.step()

        # get the Qh loss
        Qh_loss = self.compute_Qh_loss(h=h, state=state, action=action, next_state=next_state, undone=undone)
        writer.add_scalar("HJR_Qh_loss", Qh_loss, e_i)

        # update the Qh net
        self.Qh_optimizer.zero_grad()
        Qh_loss.backward()
        nn.utils.clip_grad_norm_(self.Qh_net.parameters(), 0.5)
        self.Qh_optimizer.step()

        # soft update the Qh net
        for target_param, param in zip(self.Qh_target_net.parameters(), self.Qh_net.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)

    def save_model(self, episode, map_name):
        states = {
            'Qh_net': self.Qh_net.state_dict(),
            'Vh_net': self.Vh_net.state_dict(),
            'Qh_optim': self.Qh_optimizer.state_dict(),
            'Vh_optim': self.Vh_optimizer.state_dict()
        }
        scenario_name = "all" if self.scenario_id is None else 'Scenario' + str(self.scenario_id)
        save_dir = os.path.join(self.model_path, scenario_name+"_"+map_name)
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f'model.HJR.{episode:04}.torch')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)

    def load_model(self, map_name, episode=None):
        scenario_name = "all" if self.scenario_id is None else 'Scenario' + str(self.scenario_id)
        load_dir = os.path.join(self.model_path, scenario_name+"_"+map_name)
        if episode is None:
            episode = 0
            for _, _, files in os.walk(load_dir):
                for name in files:
                    if fnmatch(name, "*torch"):
                        cur_episode = int(name.split(".")[-2])
                        if cur_episode > episode:
                            episode = cur_episode
        filepath = os.path.join(load_dir, f'model.HJR.{episode:04}.torch')
        if os.path.isfile(filepath):
            self.logger.log(f'>> Loading Safety network {self.name} from {os.path.basename(filepath)}', color="yellow")
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.Qh_net.load_state_dict(checkpoint['Qh_net'])
            self.Qh_target_net.load_state_dict(checkpoint['Qh_net'])
            self.Vh_net.load_state_dict(checkpoint['Vh_net'])
            self.Qh_optimizer.load_state_dict(checkpoint['Qh_optim'])
            self.Vh_optimizer.load_state_dict(checkpoint['Vh_optim'])
            self.continue_episode = episode
        else:
            self.logger.log(f'>> No {self.name} model found at {filepath}', 'red')
            self.continue_episode = 0
