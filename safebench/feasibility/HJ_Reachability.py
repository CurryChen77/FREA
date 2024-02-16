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
from safebench.gym_carla.net import CriticPPO, CriticTwin
from safebench.util.torch_util import CUDA, CPU


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

        self.batch_size = config['batch_size']
        self.model_path = os.path.join(config['ROOT_DIR'], config['model_path'])
        self.scenario_id = config['scenario_id']

        self.Qh_net = CUDA(CriticTwin(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))  # the Q network of constrain
        self.Qh_target_net = CUDA(CriticTwin(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))  # the Q network of constrain
        self.Qh_optimizer = optim.Adam(self.Qh_net.parameters(), lr=self.lr, eps=1e-5)  # the corresponding optimizer of Qh
        self.Qh_criterion = nn.MSELoss()

        self.Vh_net = CUDA(CriticPPO(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))  # the V network of constrain
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

    # def find_min_Qh(self, n_state):
    #     """
    #         Utilize gradient-free optimization methods to traverse the action space and find the minimum Qh value
    #     """
    #     def Qh(acceleration, steering, next_state):
    #         acceleration = torch.from_numpy(acceleration)
    #         steering = torch.from_numpy(steering)
    #         safest_action = CUDA(torch.cat((acceleration, steering), dim=1).type(next_state.dtype))
    #         Qh_value = self.Qh_net(next_state, safest_action)
    #         # the object is to min the Qh value of each state (equal to min sum of overall Qh)
    #         return Qh_value.sum().item()
    #
    #     parametrization = ng.p.Instrumentation(
    #         acceleration=ng.p.Array(shape=(self.batch_size, 1)).set_bounds(-3.0, 3.0),
    #         steering=ng.p.Array(shape=(self.batch_size, 1)).set_bounds(-0.3, 0.3),
    #         next_state=n_state
    #     )
    #
    #     optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=100)
    #     recommendation = optimizer.minimize(Qh)
    #
    #     acc = recommendation.kwargs['acceleration']
    #     steer = recommendation.kwargs['steering']
    #     acc = torch.from_numpy(acc)
    #     steer = torch.from_numpy(steer)
    #     best_action = CUDA(torch.cat((acc, steer), dim=1).type(n_state.dtype))
    #     min_Qs = self.Qh_net(n_state, best_action)
    #     return min_Qs

    @staticmethod
    def process_infos(infos):
        ego_obs = [info['ego_obs'] for info in infos]
        ego_obs = np.stack(ego_obs, axis=0)
        return ego_obs.reshape(ego_obs.shape[0], -1)

    def get_feasibility_value(self, infos):
        state = self.process_infos(infos)
        state = CUDA(torch.FloatTensor(state))
        feasibility_value = self.Vh_net(state)
        return CPU(feasibility_value)

    def get_feasibility_value_from_state(self, state):
        state = state.reshape(state.shape[0], -1)
        state = CUDA(torch.FloatTensor(state))
        feasibility_value = self.Vh_net(state)
        return CPU(feasibility_value)

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
            ego_collide = batch['ego_collide']  # if 1.0 means ego collide elif 0.0 means ego not collide
            h = torch.where(torch.isclose(ego_collide, 1.0, atol=0.01), torch.tensor(20.0), torch.tensor(-1.0))
            # h equals to threshold - min_dis, if h > 0 unsafe, else safe
            # h = torch.zeros_like(min_dis).fill_(self.min_dis_threshold) - min_dis
            # h is -1.0 when Ego is safe, else, h is 10
            # h = torch.where(min_dis < 0.5, 10.0, -1)

            del min_dis, ego_collide

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
            episode = -1
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
