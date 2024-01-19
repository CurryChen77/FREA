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
from safebench.gym_carla.net import CriticPPO, Critic
from safebench.util.torch_util import CUDA, CPU
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning)


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
        self.train_repeat_times = config['train_repeat_times']
        self.max_train_episode = config['train_episode']
        self.gamma = config['gamma']
        self.tau = config['tau']

        self.batch_size = config['batch_size']
        self.agent_info = 'ego_' + config['agent_policy_type'] + "_" + config['agent_obs_type']

        self.agent_policy_type = config['agent_policy_type']
        self.scenario_policy_type = config['scenario_policy_type']
        model_name = self.agent_policy_type + "_" + self.scenario_policy_type + "_" + "HJR"
        self.model_path = os.path.join(config['ROOT_DIR'], config['model_path'], model_name)
        self.scenario_id = config['scenario_id']

        self.Qh_net = CUDA(Critic(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))  # the Q network of constrain
        self.Qh_target_net = CUDA(Critic(dims=self.dims, state_dim=self.state_dim, action_dim=self.action_dim))  # the Q network of constrain
        self.Qh_optimizer = optim.Adam(self.Qh_net.parameters(), lr=self.lr, eps=1e-5)  # the corresponding optimizer of Qh
        self.Qh_criterion = nn.SmoothL1Loss()

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

    def find_min_Qh(self, n_state):
        """
            Utilize gradient-free optimization methods to traverse the action space and find the minimum Qh value
        """
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

    @staticmethod
    def soft_update(target_net, current_net, tau=5e-3):
        """soft update target network via current network

        target_net: update target network via current network to make training more stable.
        current_net: current network update via an optimizer
        tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    @staticmethod
    def safe_expectile_loss(diff, expectile=0.8):
        weight = torch.where(diff < 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def compute_Vh_loss(self, state, action):
        Qh = self.Qh_target_net(state, action)
        Qh_max = torch.max(Qh)  # the Qh is about the constraint h, so lower means better, this is different from the reward, higher the better
        Vh = self.Vh_net(state)

        Vh_loss = self.safe_expectile_loss(Qh_max - Vh).mean()
        return Vh_loss

    def compute_Qh_loss(self, h, state, action, next_state, undone):
        next_Vh = self.Vh_net(next_state)
        Qh = self.Qh_net(state, action)

        Qh_nonterminal = (1. - self.gamma) * h + self.gamma * torch.maximum(h, next_Vh)
        target_Qh = Qh_nonterminal * undone + h * (1. - undone)
        Qh_loss = self.Qh_criterion(Qh, target_Qh)
        return Qh_loss

    def train(self, buffer, writer, e_i):

        with torch.no_grad():
            # learning rate decay
            self.lr_decay(e_i)  # add the learning rate decay

            batch = buffer.get()

            states = CUDA(torch.FloatTensor(batch['obs']))
            next_states = CUDA(torch.FloatTensor(batch['next_obs']))
            actions = CUDA(torch.FloatTensor(batch['actions']))
            undones = CUDA(torch.FloatTensor(1-batch['dones']))
            # the ego min distance from the infos
            min_dis = CUDA(torch.FloatTensor(batch['constraint_h']))
            # h = threshold - min_dis, if h > 0 unsafe, else safe
            hs = torch.zeros_like(min_dis).fill_(self.min_dis_threshold) - min_dis
            buffer_size = states.shape[0]

            del min_dis

        # start to train, use gradient descent without batch size
        update_times = int(buffer_size * self.train_repeat_times / self.batch_size)
        assert update_times >= 1

        for _ in range(update_times):
            indices = torch.randint(buffer_size, size=(self.batch_size,), requires_grad=False)
            state = states[indices]
            next_state = next_states[indices]
            action = actions[indices]
            h = hs[indices]
            undone = undones[indices]

            # get the Vh loss
            Vh_loss = self.compute_Vh_loss(state=state, action=action)
            writer.add_scalar("HJR_Vh_loss", Vh_loss, e_i)
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

        # reset buffer
        buffer.reset_buffer()

    def save_model(self, episode, map_name):
        states = {
            'Qh_net': self.Qh_net.state_dict(),
            'Vh_net': self.Vh_net.state_dict(),
            'Qh_optim': self.Qh_optimizer.state_dict(),
            'Vh_optim': self.Vh_optimizer.state_dict()
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
