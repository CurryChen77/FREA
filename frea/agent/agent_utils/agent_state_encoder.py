#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：agent_state_encoder.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/22
"""


import carla
import torch
from torch.nn.utils.rnn import pad_sequence
from frea.agent.agent_utils.coordinate_utils import normalize_angle
from frea.agent.PlanT.dataset import split_large_BB, generate_batch
from frea.agent.agent_utils.explainability_utils import *
from frea.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from einops import rearrange
from pytorch_lightning.utilities.cloud_io import load as pl_load
import numpy as np
from rdp import rdp
import torch.nn as nn
import os.path as osp

from transformers import (
    AutoConfig,
    AutoModel,
)


class AgentStateEncoder(object):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.obs_type = config['obs_type']
        self.viz_attn_map = config['viz_attn_map']
        self.model_path = config['pretrained_model_path']
        self.net = EncoderModel(config)

    def load_ckpt(self, strict=True):
        updated_model_path = self.config['pretrained_model_path']
        checkpoint = pl_load(updated_model_path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(checkpoint["state_dict"], strict=strict)
        self.world = CarlaDataProvider.get_world()
        self.logger.log('>> Loading the pretrained ego state encoder.', color='yellow')

    def get_most_relevant_vehicle(self, attn_vector, topk=1):
        # get ids of all vehicles in detection range
        data_car_ids = [
            float(x['id'])
            for x in self.data if x['class'] == 'Car']

        # get topk indices of attn_vector
        if topk > len(attn_vector):
            topk = len(attn_vector)
        else:
            topk = topk

        attn_indices = np.argpartition(attn_vector, -topk)[-topk:]

        # get carla vehicles ids of topk vehicles
        keep_vehicle_ids = []
        keep_vehicle_attn = []
        for indice in attn_indices:
            if indice < len(data_car_ids):
                keep_vehicle_ids.append(data_car_ids[indice])
                keep_vehicle_attn.append(attn_vector[indice])

        # if we don't have any detected vehicle we should not have any ids here
        # otherwise we want #topk vehicles
        if len(self.data_car) > 0:
            assert len(keep_vehicle_ids) == topk
        else:
            assert len(keep_vehicle_ids) == 0

        # get topk (top 1) vehicles indices
        if len(keep_vehicle_ids) >= 1:
            most_relevant_vehicle_id = keep_vehicle_ids[0]
            most_relevant_vehicle = CarlaDataProvider.get_actor_by_id(most_relevant_vehicle_id)
        else:
            most_relevant_vehicle = None

        return most_relevant_vehicle

    @torch.no_grad()
    def get_encoded_state(self, ego_vehicle, ego_nearby_vehicles, waypoints, traffic_light_hazard):
        if len(waypoints) > 1:
            target_point = waypoints[1]  # the preview waypoint
        else:
            target_point = waypoints[0]  # the first waypoint
        label_raw = self.get_bev_boxes(ego_vehicle, ego_nearby_vehicles, waypoints)
        input_batch = self.get_input_batch(label_raw, target_point, traffic_light_hazard)
        x, y, _, tp, light = input_batch
        encoded_state, attn_map = self.net(x, y, target_point=tp, light_hazard=light)

        # get the most relevant vehicle as the controlled background vehicle
        attn_vector = get_attn_norm_vehicles(self.config['attention_score'], self.data_car, attn_map)
        most_relevant_vehicle = self.get_most_relevant_vehicle(attn_vector, topk=1)

        if self.viz_attn_map:
            keep_vehicle_ids, attn_indices, keep_vehicle_attn = get_vehicleID_from_attn_scores(self.data, self.data_car, self.config['topk'], attn_vector)
            draw_attention_bb_in_carla(self.world, keep_vehicle_ids, keep_vehicle_attn)
        return most_relevant_vehicle

    def get_bev_boxes(self, ego_veh, ego_nearby_vehicles, waypoints):
        """
            modify from the PlanT
        """
        # -----------------------------------------------------------
        # Ego vehicle
        # -----------------------------------------------------------

        # add vehicle velocity and brake flag
        ego_transform = CarlaDataProvider.get_transform(ego_veh)
        ego_control = ego_veh.get_control()
        ego_velocity = ego_veh.get_velocity()
        ego_speed = self.get_forward_speed(transform=ego_transform, velocity=ego_velocity)  # In m/s
        ego_brake = ego_control.brake
        ego_rotation = ego_transform.rotation
        ego_matrix = np.array(ego_transform.get_matrix())
        ego_extent = ego_veh.bounding_box.extent
        ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z]) * 2.
        ego_yaw = ego_rotation.yaw / 180 * np.pi
        relative_yaw = 0
        relative_pos = self.get_relative_transform(ego_matrix, ego_matrix)

        results = []

        # add ego-vehicle to results list
        # the format is category, extent*3, position*3, yaw, points_in_bbox, distance, id
        # the position is in lidar coordinates
        result = {"class": "Car",
                  "extent": [ego_dx[2], ego_dx[0], ego_dx[1]],  # TODO:
                  "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                  "yaw": relative_yaw,
                  "num_points": -1,
                  "distance": -1,
                  "speed": ego_speed,
                  "brake": ego_brake,
                  "id": int(ego_veh.id),
                  }
        results.append(result)

        # -----------------------------------------------------------
        # Other vehicles
        # -----------------------------------------------------------

        for vehicle in ego_nearby_vehicles:
            vehicle_transform = CarlaDataProvider.get_transform(vehicle)
            vehicle_rotation = vehicle_transform.rotation

            vehicle_matrix = np.array(vehicle_transform.get_matrix())

            vehicle_extent = vehicle.bounding_box.extent
            dx = np.array([vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]) * 2.
            yaw = vehicle_rotation.yaw / 180 * np.pi

            relative_yaw = normalize_angle(yaw - ego_yaw)
            relative_pos = self.get_relative_transform(ego_matrix, vehicle_matrix)

            vehicle_control = vehicle.get_control()
            vehicle_velocity = vehicle.get_velocity()
            vehicle_speed = self.get_forward_speed(transform=vehicle_transform, velocity=vehicle_velocity)  # In m/s
            vehicle_brake = vehicle_control.brake

            # # filter bbox that didn't contain points of contains fewer points
            # if not lidar is None:
            #     num_in_bbox_points = self.get_points_in_bbox(ego_matrix, vehicle_matrix, dx, lidar)
            #     # print("num points in bbox", num_in_bbox_points)
            # else:
            #     num_in_bbox_points = -1
            num_in_bbox_points = -1

            distance = np.linalg.norm(relative_pos)

            result = {
                "class": "Car",
                "extent": [dx[2], dx[0], dx[1]],  # TODO
                "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                "yaw": relative_yaw,
                "num_points": int(num_in_bbox_points),
                "distance": distance,
                "speed": vehicle_speed,
                "brake": vehicle_brake,
                "id": int(vehicle.id),
            }
            results.append(result)

        # -----------------------------------------------------------
        # Route rdp
        # -----------------------------------------------------------
        max_len = 50
        waypoint_route = np.array([[node[0], node[1]] for node in waypoints])
        if len(waypoint_route) < max_len:
            max_len = len(waypoint_route)
        shortened_route = rdp(waypoint_route[:max_len], epsilon=0.5)

        # convert points to vectors
        vectors = shortened_route[1:] - shortened_route[:-1]
        midpoints = shortened_route[:-1] + vectors / 2.
        norms = np.linalg.norm(vectors, axis=1)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])

        for i, midpoint in enumerate(midpoints):
            # find distance to center of waypoint
            center_bounding_box = carla.Location(midpoint[0], midpoint[1], 0.0)
            transform = carla.Transform(center_bounding_box)
            route_matrix = np.array(transform.get_matrix())
            relative_pos = self.get_relative_transform(ego_matrix, route_matrix)
            distance = np.linalg.norm(relative_pos)

            # find distance to beginning of bounding box
            starting_bounding_box = carla.Location(shortened_route[i][0], shortened_route[i][1], 0.0)
            st_transform = carla.Transform(starting_bounding_box)
            st_route_matrix = np.array(st_transform.get_matrix())
            st_relative_pos = self.get_relative_transform(ego_matrix, st_route_matrix)
            st_distance = np.linalg.norm(st_relative_pos)

            # only store route boxes that are near the ego vehicle
            if i > 0 and st_distance > 30:
                continue

            length_bounding_box = carla.Vector3D(norms[i] / 2., ego_extent.y, ego_extent.z)
            bounding_box = carla.BoundingBox(transform.location, length_bounding_box)
            bounding_box.rotation = carla.Rotation(pitch=0.0,
                                                   yaw=angles[i] * 180 / np.pi,
                                                   roll=0.0)

            route_extent = bounding_box.extent
            dx = np.array([route_extent.x, route_extent.y, route_extent.z]) * 2.
            relative_yaw = normalize_angle(angles[i] - ego_yaw)

            # visualize subsampled route
            # self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1,
            #                             color=carla.Color(0, 255, 255, 255), life_time=(10.0/self.frame_rate_sim))

            result = {
                "class": "Route",
                "extent": [dx[2], dx[0], dx[1]],  # TODO
                "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                "yaw": relative_yaw,
                "centre_distance": distance,
                "starting_distance": st_distance,
                "id": i,
            }
            results.append(result)

        return results

    def get_forward_speed(self, transform, velocity):
        """ Convert the vehicle transform directly to forward speed """

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def get_relative_transform(self, ego_matrix, vehicle_matrix):
        """
        return the relative transform from ego_pose to vehicle pose
        """
        relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
        rot = ego_matrix[:3, :3].T
        relative_pos = rot @ relative_pos

        # transform to right-handed system
        relative_pos[1] = - relative_pos[1]

        # transform relative pos to virtual lidar system
        rot = np.eye(3)
        trans = - np.array([1.3, 0.0, 2.5])
        relative_pos = rot @ relative_pos + trans

        return relative_pos

    def get_input_batch(self, label_raw, target_point, traffic_light_hazard):
        sample = {'input': [], 'output': [], 'brake': [], 'waypoints': [], 'target_point': [], 'light': []}

        data = label_raw[1:]  # remove first element (ego vehicle)

        data_car = [[
            1.,  # type indicator for cars
            float(x['position'][0]) - float(label_raw[0]['position'][0]),
            float(x['position'][1]) - float(label_raw[0]['position'][1]),
            float(x['yaw'] * 180 / 3.14159265359),  # in degrees
            float(x['speed'] * 3.6),  # in km/h
            float(x['extent'][2]),
            float(x['extent'][1]),
        ] for x in data if x['class'] == 'Car']
        # if we use the far_node as target waypoint we need the route as input
        data_route = [
            [
                2.,  # type indicator for route
                float(x['position'][0]) - float(label_raw[0]['position'][0]),
                float(x['position'][1]) - float(label_raw[0]['position'][1]),
                float(x['yaw'] * 180 / 3.14159265359),  # in degrees
                float(x['id']),
                float(x['extent'][2]),
                float(x['extent'][1]),
            ]
            for j, x in enumerate(data)
            if x['class'] == 'Route'
               and float(x['id']) < self.config['training']['max_NextRouteBBs']]

        # we split route segment longer than 10m into multiple segments
        # improves generalization
        data_route_split = []
        for route in data_route:
            if route[6] > 10:
                routes = split_large_BB(route, len(data_route_split))
                data_route_split.extend(routes)
            else:
                data_route_split.append(route)

        data_route = data_route_split[:self.config['training']['max_NextRouteBBs']]

        assert len(data_route) <= self.config['training']['max_NextRouteBBs'], 'Too many routes'

        features = data_car + data_route

        sample['input'] = features

        # dummy data
        sample['output'] = features
        sample['light'] = traffic_light_hazard

        local_command_point = np.array([target_point[0], target_point[1]])
        sample['target_point'] = local_command_point

        batch = [sample]

        input_batch = generate_batch(batch)

        self.data = data
        self.data_car = data_car
        self.data_route = data_route

        return input_batch


class EncoderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config_all = config
        self.config_net = config['network']

        self.object_types = 2 + 1  # vehicles, route +1 for padding and wp embedding
        self.num_attributes = 6  # x,y,yaw,speed/id, extent x, extent y

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
        self.drop = nn.Dropout(self.config_net['embd_pdrop'])

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

    def forward(self, idx, target=None, target_point=None, light_hazard=None):

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

        return x, attn_map