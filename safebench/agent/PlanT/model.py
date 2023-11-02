#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：model.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/22
@source  ：This file is modified from <https://github.com/autonomousvision/plant/tree/1bfb695910d816e70f53521aa263648072edea8e>
"""

import logging
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from einops import rearrange

from transformers import (
    AutoConfig,
    AutoModel,
)


class HFLM(nn.Module):
    def __init__(self, config_net, config_all):
        super().__init__()
        self.config_all = config_all
        self.config_net = config_net

        self.object_types = 2 + 1  # vehicles, route +1 for padding and wp embedding
        self.num_attributes = 6  # x,y,yaw,speed/id, extent x, extent y

        precisions = [
            self.config_all['pre_training'].get("precision_pos", 4),
            self.config_all['pre_training'].get("precision_pos", 4),
            self.config_all['pre_training'].get("precision_angle", 4),
            self.config_all['pre_training'].get("precision_speed", 4),
            self.config_all['pre_training'].get("precision_pos", 4),
            self.config_all['pre_training'].get("precision_pos", 4),
        ]

        self.vocab_size = [2**i for i in precisions]

        # model
        config = AutoConfig.from_pretrained(
            self.config_net['hf_checkpoint']
        )  # load config from hugging face model
        n_embd = config.hidden_size
        self.model = AutoModel.from_config(config=config)

        # sequence padding for batching
        self.cls_emb = nn.Parameter(
            torch.randn(1, self.num_attributes + 1)
        )  # +1 because at this step we still have the type indicator
        self.eos_emb = nn.Parameter(
            torch.randn(1, self.num_attributes + 1)
        )  # unnecessary TODO: remove

        # token embedding
        self.tok_emb = nn.Linear(self.num_attributes, n_embd)
        # object type embedding
        self.obj_token = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, self.num_attributes))
                for _ in range(self.object_types)
            ]
        )
        self.obj_emb = nn.ModuleList(
            [nn.Linear(self.num_attributes, n_embd) for _ in range(self.object_types)]
        )
        self.drop = nn.Dropout(config_net['embd_pdrop'])

        # decoder head forecasting
        if (
            self.config_all['pre_training'].get("pretraining", "none") == "forecast"
            or self.config_all['training'].get("pretraining_path", "none") != "none"
        ):
            # one head for each attribute type -> we have different precision per attribute
            self.heads = nn.ModuleList(
                [
                    nn.Linear(n_embd, self.vocab_size[i])
                    for i in range(self.num_attributes)
                ]
            )

        # wp (CLS) decoding
        self.wp_head = nn.Linear(n_embd, 64)
        self.wp_decoder = nn.GRUCell(input_size=4, hidden_size=65)
        self.wp_relu = nn.ReLU()
        self.wp_output = nn.Linear(65, 2)

        # PID controller
        self.turn_controller = PIDController(K_P=0.9, K_I=0.75, K_D=0.3, n=20)
        self.speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=20)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = torch.nn.Linear
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith("_ih") or pn.endswith("_hh"):
                    # all recurrent weights will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("_emb") or "_token" in pn:
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    def forward(self, idx, target=None, target_point=None, light_hazard=None):

        if self.config_all['pre_training'].get("pretraining", "none") == "none":
            assert (
                target_point is not None
            ), "target_point must be provided for wp output"
            assert (
                light_hazard is not None
            ), "light_hazard must be provided for wp output"

        # create batch of same size as input
        x_batched = torch.cat(idx, dim=0)
        input_batch = self.pad_sequence_batch(x_batched)
        input_batch_type = input_batch[:, :, 0]  # car or map
        input_batch_data = input_batch[:, :, 1:]

        # create same for output in case of multitask training to use this as ground truth
        if target is not None:
            y_batched = torch.cat(target, dim=0)
            output_batch = self.pad_sequence_batch(y_batched)
            output_batch_type = output_batch[:, :, 0]  # car or map
            output_batch_data = output_batch[:, :, 1:]

        # create masks by object type
        car_mask = (input_batch_type == 1).unsqueeze(-1)
        route_mask = (input_batch_type == 2).unsqueeze(-1)
        other_mask = torch.logical_and(route_mask.logical_not(), car_mask.logical_not())
        masks = [car_mask, route_mask, other_mask]

        # get size of input
        (B, O, A) = (input_batch_data.shape)  # batch size, number of objects, number of attributes

        # embed tokens object wise (one object -> one token embedding)
        input_batch_data = rearrange(
            input_batch_data, "b objects attributes -> (b objects) attributes"
        )
        embedding = self.tok_emb(input_batch_data)
        embedding = rearrange(embedding, "(b o) features -> b o features", b=B, o=O)

        # create object type embedding
        obj_embeddings = [
            self.obj_emb[i](self.obj_token[i]) for i in range(self.object_types)
        ]  # list of a tensors of size 1 x features

        # add object type embedding to embedding (mask needed to only add to the correct tokens)
        embedding = [
            (embedding + obj_embeddings[i]) * masks[i] for i in range(self.object_types)
        ]
        embedding = torch.sum(torch.stack(embedding, dim=1), dim=1)

        # embedding dropout
        x = self.drop(embedding)

        # Transformer Encoder; use embedding for hugging face model and get output states and attention map
        output = self.model(**{"inputs_embeds": embedding}, output_attentions=True)
        x, attn_map = output.last_hidden_state, output.attentions

        # forecasting encoding
        if (
            self.config_all['pre_training'].get("pretraining", "none") == "forecast"
            or self.config_all['training'].get("pretraining_path", "none") != "none"
        ):
            car_mask_output = (output_batch_type == 1).unsqueeze(-1)
            non_car_mask_output = (output_batch_type != 1).unsqueeze(-1)
            # size: list of self.num_attributes tensors of size B x O x vocab_size (vocab_size differs for each attribute)
            # we do forecasting only for vehicles
            logits = [
                self.heads[i](x) * car_mask_output - 999 * non_car_mask_output
                for i in range(self.num_attributes)
            ]
            logits = [
                rearrange(logit, "b o vocab_size -> (b o) vocab_size")
                for logit in logits
            ]

            # get target (GT) in same shape as logits
            targets = [
                output_batch_data[:, :, i].unsqueeze(-1) * car_mask_output - 999 * non_car_mask_output
                for i in range(self.num_attributes)
            ]
            targets = [
                rearrange(target, "b o vocab_size -> (b o) vocab_size").long()
                for target in targets
            ]

            # if we do pre-training (not multitask) we don't need wp for pre-training step, so we can return here
            if (
                self.config_all['pre_training'].get("pretraining", "none") == "forecast"
                and self.config_all['pre_training'].get("multitask", False) == False
            ):
                return logits, targets
        else:
            logits = None

        # get waypoint predictions
        z = self.wp_head(x[:, 0, :])
        # add traffic ligth flag
        z = torch.cat((z, light_hazard), 1)

        output_wp = list()

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype)
        x = x.type_as(z)

        # autoregressive generation of output waypoints
        for _ in range(self.config_all['training']['pred_len']):
            x_in = torch.cat([x, target_point], dim=1)
            z = self.wp_decoder(x_in, z)
            dx = self.wp_output(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)

        # pred the wapoints in the vehicle coordinate and we convert it to lidar coordinate here because the GT waypoints is in lidar coordinate
        pred_wp[:, :, 0] = pred_wp[:, :, 0] - 1.3

        if (
            self.config_all['pre_training'].get("pretraining", "none") == "forecast"
            and self.config_all['pre_training'].get("multitask", False) == True
        ):
            return logits, targets, pred_wp, attn_map
        else:
            return logits, targets, pred_wp, attn_map

    def pad_sequence_batch(self, x_batched):
        """
        Pads a batch of sequences to the longest sequence in the batch.
        """
        # split input into components
        x_batch_ids = x_batched[:, 0]

        x_tokens = x_batched[:, 1:]

        B = int(x_batch_ids[-1].item()) + 1
        input_batch = []
        for batch_id in range(B):
            # get the batch of elements
            x_batch_id_mask = x_batch_ids == batch_id

            # get the batch of types
            x_tokens_batch = x_tokens[x_batch_id_mask]

            x_seq = torch.cat([self.cls_emb, x_tokens_batch, self.eos_emb], dim=0)

            input_batch.append(x_seq)

        padded = torch.swapaxes(pad_sequence(input_batch), 0, 1)
        input_batch = padded[:B]

        return input_batch

    def control_pid(self, waypoints, velocity, is_stuck=False):
        """Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): output of self.plan()
            velocity (tensor): speedometer input
        """
        assert waypoints.size(0) == 1
        waypoints = waypoints[0].data.cpu().numpy()
        # when training we transform the waypoints to lidar coordinate, so we need to change is back when control
        waypoints[:, 0] += 1.3
        speed = velocity[0].data.cpu().numpy()

        pre_difined_desired_speed = CarlaDataProvider.get_ego_desired_speed()
        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * pre_difined_desired_speed//2  # near the desired speed
        if is_stuck:
            desired_speed = np.array(pre_difined_desired_speed)

        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1

        delta = np.clip(desired_speed - speed, 0.0, 0.25)

        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0
        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90
        if brake:
            angle = 0.0
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        return steer, throttle, brake


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative
