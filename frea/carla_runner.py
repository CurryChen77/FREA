#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：carla_runner.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""

import copy
import os
import re
import numpy as np
import carla
import pygame
from frea.feasibility import FEASIBILITY_LIST
from tqdm import tqdm
import os.path as osp
from torch.utils.tensorboard import SummaryWriter

from frea.util.run_util import load_config
from frea.gym_carla.env_wrapper import VectorWrapper
from frea.gym_carla.envs.render import BirdeyeRender
from frea.gym_carla.replay_buffer import ReplayBuffer
from frea.gym_carla.rollout_buffer import RolloutBuffer

from frea.agent import AGENT_POLICY_LIST
from frea.scenario import SCENARIO_POLICY_LIST
from frea.agent.agent_utils.agent_state_encoder import AgentStateEncoder

from frea.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from frea.scenario.scenario_data_loader import ScenarioDataLoader
from frea.scenario.tools.scenario_utils import scenario_parse

from frea.util.logger import Logger, setup_logger_kwargs
from frea.util.metric_util import get_route_scores


class CarlaRunner:
    def __init__(self, agent_config, scenario_config, feasibility_config):
        self.scenario_config = scenario_config
        self.agent_config = agent_config
        self.feasibility_config = feasibility_config
        self.current_map = None
        self.all_map = re.findall(r'Town\d+', scenario_config['scenario_type'])
        self.birdeye_render = None
        self.display = None

        self.seed = scenario_config['seed']
        self.output_dir = scenario_config['output_dir']
        self.mode = scenario_config['mode']
        self.save_video = scenario_config['save_video']

        self.eval_mode = scenario_config['eval_mode']
        self.viz_route = scenario_config['viz_route']
        self.num_scenario = scenario_config['num_scenario']  # default 2
        self.fixed_delta_seconds = scenario_config['fixed_delta_seconds']
        self.CBV_selection = scenario_config['CBV_selection']
        self.scenario_agent_learnable = scenario_config['learnable']
        self.scenario_id = scenario_config['scenario_id']
        # if the scenario agent need feasibility
        self.use_feasibility = scenario_config['feasibility']
        self.scenario_policy_type = scenario_config['policy_type']
        self.agent_policy_type = agent_config['policy_type']

        # apply settings to carla
        self.client = carla.Client('localhost', scenario_config['port'])
        self.client.set_timeout(15.0)
        self.world = None
        self.env = None

        self.env_params = {
            'mode': self.mode,  # the mode of the script
            'eval_mode': self.eval_mode,  # the mode of the evaluation
            'search_radius': 25,  # the default search radius
            'traffic_intensity': 0.3 if self.mode == 'eval' else 0.6,  # the default traffic intensity
            'goal_point_radius': 2,  # the default goal point radius
            'auto_ego': scenario_config['auto_ego'],
            'viz_route': self.viz_route,  # whether to visualize the route
            'ego_agent_learnable': agent_config['learnable'],  # whether the ego agent is a learnable method
            'scenario_agent_learnable': scenario_config['learnable'],  # whether the scenario agent is a learnable method
            'agent_obs_type': agent_config['obs_type'],  # default 0 (only 4 dimensions states )
            'CBV_selection': self.CBV_selection,  # the method using for selection the controlled bv
            'ROOT_DIR': scenario_config['ROOT_DIR'],
            'signalized_junction': False,  # whether the signal controls the junction
            'warm_up_steps': 4,  # number of ticks after spawning the vehicles
            'disable_lidar': True,  # show bird-eye view lidar or not
            'enable_sem': False,  # whether to enable the semantic camera
            'display_size': 512,  # screen size of one bird-eye view window
            'obs_range': 60,  # observation range (meter)
            'd_behind': 16,  # distance behind the ego vehicle (meter)
            'max_past_step': 1,  # the number of past steps to draw
            'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
            'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
            'max_episode_step': 300,  # maximum time steps per episode
            'max_waypt': 12,  # maximum number of waypoints
            'lidar_bin': 0.0625,  # bin size of lidar sensor (meter)
            'out_lane_thres': 4,  # threshold for out of lane (meter)
            'desired_speed': 6,  # desired speed (m/s)
        }

        # pass config from scenario to agent
        agent_config['mode'] = self.mode
        agent_config['desired_speed'] = self.env_params['desired_speed']
        agent_config['num_scenario'] = scenario_config['num_scenario']
        agent_config['scenario_id'] = self.scenario_id
        agent_config['scenario_policy_type'] = scenario_config['policy_type']

        # pass config from agent, scenario to feasibility
        feasibility_config['agent_policy_type'] = agent_config['policy_type']
        feasibility_config['scenario_policy_type'] = scenario_config['policy_type']
        feasibility_config['scenario_id'] = self.scenario_id
        feasibility_config['agent_obs_type'] = agent_config['obs_type']
        feasibility_config['agent_action_dim'] = agent_config['ego_action_dim']
        feasibility_config['search_radius'] = self.env_params['search_radius']
        feasibility_config['ego_agent_learnable'] = agent_config['learnable']

        # pass config from agent to scenario
        scenario_config['agent_policy'] = agent_config['policy_type']
        scenario_config['agent_obs_type'] = agent_config['obs_type']

        CarlaDataProvider.set_ego_desired_speed(self.env_params['desired_speed'])

        # define logger
        logger_kwargs = setup_logger_kwargs(
            self.output_dir,
            self.seed,
            self.mode,
            agent=agent_config['policy_type'],
            scenario=scenario_config['policy_type'],
            CBV_selection=self.CBV_selection,
            all_map_name=self.all_map,
            eval_obj=agent_config['eval_obj'],
            pretrained_ego=agent_config['pretrain_ego'],
            pretrained_cbv=agent_config['pretrain_cbv']
        )
        self.logger = Logger(**logger_kwargs)

        # prepare parameters
        if self.mode == 'train_agent':
            self.buffer_capacity = agent_config['buffer_capacity']
            self.save_freq = agent_config['save_freq']
            self.train_episode_list = agent_config['train_episode']
            self.logger.save_config(agent_config)
        elif self.mode == 'train_scenario':
            self.buffer_capacity = scenario_config['buffer_capacity']
            self.save_freq = scenario_config['save_freq']
            self.train_episode_list = scenario_config['train_episode']
            self.logger.save_config(scenario_config)
        elif self.mode == 'collect_feasibility_data':
            self.buffer_capacity = feasibility_config['buffer_capacity']
            self.feasibility_data_path = feasibility_config['data_path']
            self.logger.save_config(feasibility_config)
        elif self.mode == 'eval':
            if self.eval_mode == 'analysis':
                self.save_freq = scenario_config['save_freq']
                self.logger.log('>> Evaluation Mode, analyzing result', 'yellow')
                self.logger.create_eval_dir(load_existing_results=True, scenario_id=self.scenario_config['scenario_id'])
            else:
                self.logger.log('>> Evaluation Mode, rendering result', 'yellow')
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}.")

        # define the ego state encoder if needed
        self.agent_state_encoder = None
        state_encoder_config = None
        # if the CBV selection method is based on attention
        if self.CBV_selection == 'attention-based':
            # initial the agent state encoder
            root_path = agent_config['ROOT_DIR']
            state_encoder_path = osp.join(root_path, 'frea/agent/config/state_encoder.yaml')
            state_encoder_config = load_config(state_encoder_path)
            state_encoder_config['viz_attn_map'] = True if self.mode == 'eval' else None  # viz the attention map when eval
            self.agent_state_encoder = AgentStateEncoder(state_encoder_config, self.logger)

        # define agent and scenario
        self.logger.log('>> Mode: ' + self.mode, color="yellow")
        self.logger.log('>> Agent Policy: ' + agent_config['policy_type'], color="yellow")
        self.logger.log('>> Scenario Policy: ' + scenario_config['policy_type'], color="yellow")
        if self.use_feasibility:
            self.logger.log('>> Feasibility Policy: ' + feasibility_config['type'], color="yellow")
        if self.agent_state_encoder:
            self.logger.log('>> Using state encoder: ' + state_encoder_config['obs_type'], color="yellow")

        if self.scenario_config['auto_ego']:
            self.logger.log('>> Using auto-polit for ego vehicle, action of policy will be ignored', 'yellow')

        self.logger.log('>> ' + '-' * 40)

        # define agent policy, scenario policy (feasibility policy)
        self.agent_policy = AGENT_POLICY_LIST[agent_config['policy_type']](agent_config, logger=self.logger)
        self.scenario_policy = SCENARIO_POLICY_LIST[scenario_config['policy_type']](scenario_config, logger=self.logger)
        self.feasibility_policy = FEASIBILITY_LIST[feasibility_config['type']](feasibility_config, logger=self.logger) if self.use_feasibility else None

        if self.save_video:
            assert self.eval_mode == 'render', "only allow video saving in eval mode"
            self.logger.init_video_recorder()

    def _init_world(self, town):
        self.logger.log(f">> Initializing carla world: {town}")
        self.world = self.client.load_world(town)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(self.scenario_config['tm_port'])
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

    def _init_renderer(self):
        self.logger.log(">> Initializing pygame birdeye renderer")
        pygame.init()
        flag = pygame.HWSURFACE | pygame.DOUBLEBUF
        # flag = flag | pygame.HIDDEN  # not showing the pygame window on the screen

        # [bird-eye view, Lidar, front view] or [bird-eye view, front view]
        if self.env_params['disable_lidar']:
            if self.env_params['enable_sem']:
                window_size = (self.env_params['display_size'] * 3, self.env_params['display_size'] * self.num_scenario)
            else:
                window_size = (self.env_params['display_size'] * 2, self.env_params['display_size'] * self.num_scenario)
        else:
            if self.env_params['enable_sem']:
                window_size = (self.env_params['display_size'] * 4, self.env_params['display_size'] * self.num_scenario)
            else:
                window_size = (self.env_params['display_size'] * 3, self.env_params['display_size'] * self.num_scenario)

        self.display = pygame.display.set_mode(window_size, flag)

        # initialize the render for generating observation and visualization
        pixels_per_meter = self.env_params['display_size'] / self.env_params['obs_range']
        pixels_ahead_vehicle = (self.env_params['obs_range'] / 2 - self.env_params['d_behind']) * pixels_per_meter
        self.birdeye_params = {
            'screen_size': [self.env_params['display_size'], self.env_params['display_size']],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle,
        }
        self.birdeye_render = BirdeyeRender(self.world, self.birdeye_params, logger=self.logger)

    def train(self, data_loader, start_episode=0):
        # create the tensorboard writer
        log_dir = self.logger.output_dir
        writer_dir = osp.join(log_dir, "Scenario" + str(self.scenario_id) + '_' + self.current_map)
        writer = SummaryWriter(log_dir=writer_dir)

        # general buffer for both agent and scenario
        buffer, onpolicy = self.check_onpolicy(start_episode=start_episode)

        data_loader.set_mode("train")

        for e_i in tqdm(range(start_episode, self.train_episode_list[self.current_map] + 1)):
            # sample scenarios in this town (one town could contain multiple scenarios)
            # simulate multiple scenarios in parallel (usually 2 scenarios)
            sampled_scenario_configs, config_lengths = data_loader.sampler()
            # if config length < num_scenarios, needs to balance to storing placement within buffer
            if self.mode == 'train_agent' and self.agent_policy.type == 'onpolicy':
                buffer.check_scenario_id_for_saving(config_lengths)
            # reset the index counter to create endless loader
            data_loader.reset_idx_counter()

            obs, infos = self.env.reset(sampled_scenario_configs)

            # get ego vehicle from scenario
            self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos)

            # start loop
            agent_episode_reward = []
            scenario_episode_reward = []
            while not self.env.all_scenario_done():
                # get action from agent policy and scenario policy (assume using one batch)
                ego_actions = self.agent_policy.get_action(obs, infos, deterministic=False)
                scenario_actions = self.scenario_policy.get_action(obs, infos, deterministic=False)

                # apply action to env and get obs
                next_obs, next_transition_obs, rewards, dones, next_infos, next_transition_infos = self.env.step(ego_actions, scenario_actions, onpolicy)
                # store to the replay buffer
                buffer.store([ego_actions, scenario_actions, obs, next_obs, rewards, dones], additional_dict=[infos, next_infos])
                # for transition
                infos = next_transition_infos
                obs = copy.deepcopy(next_transition_obs)

                if self.mode == 'train_agent':
                    agent_episode_reward.append(np.mean(rewards))
                if self.mode == 'train_scenario':
                    scenario_reward = np.concatenate([np.array(list(info['CBVs_reward'].values())) for info in next_infos])
                    if scenario_reward.size != 0:
                        scenario_episode_reward.append(np.mean(scenario_reward))  # the mean reward across CBVs across scenarios at this moment

                # train off-policy agent or scenario
                if self.mode == 'train_agent' and self.agent_policy.type == 'offpolicy':
                    self.agent_policy.train(buffer, writer, e_i)
                elif self.mode == 'train_scenario' and self.scenario_policy.type == 'offpolicy':
                    self.scenario_policy.train(buffer, writer, e_i)

            self.logger.log('>> Start Cleaning', 'yellow')
            # end up environment
            self.env.clean_up()

            if self.mode == 'train_agent':
                writer.add_scalar("Agent_episode_reward", np.sum(agent_episode_reward), e_i)
            if self.mode == 'train_scenario':
                writer.add_scalar("Scenario_average_reward_per_step", np.mean(scenario_episode_reward), e_i)
                writer.add_scalar("Scenario_episode_reward", np.sum(scenario_episode_reward), e_i)

            # train on-policy agent or scenario
            if self.mode == 'train_agent' and self.agent_policy.type == 'onpolicy':
                self.agent_policy.train(buffer, writer, e_i) if e_i != start_episode and all(buffer.agent_full) else None
            elif self.mode == 'train_scenario' and self.scenario_policy.type == 'onpolicy':
                self.scenario_policy.train(buffer, writer, e_i) if e_i != start_episode and buffer.scenario_full else None

            # save checkpoints
            if e_i != start_episode and e_i % self.save_freq == 0:
                if self.mode == 'train_agent':
                    self.agent_policy.save_model(e_i, map_name=self.current_map, buffer=buffer)
                if self.mode == 'train_scenario':
                    self.scenario_policy.save_model(e_i, map_name=self.current_map, buffer=buffer)

        # close the tensorboard writer
        writer.close()

    def eval(self, data_loader):
        num_finished_scenario = 0
        data_loader.set_mode("eval")
        data_loader.reset_idx_counter()

        _, onpolicy = self.check_onpolicy()

        while len(data_loader) > 0:
            # sample scenarios
            sampled_scenario_configs, num_sampled_scenario = data_loader.sampler()
            num_finished_scenario += num_sampled_scenario

            obs, infos = self.env.reset(sampled_scenario_configs)

            # get ego vehicle from scenario
            self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos)

            score_list = {s_i: [] for s_i in range(num_sampled_scenario)}
            while not self.env.all_scenario_done():
                # get action from agent policy and scenario policy (assume using one batch)
                ego_actions = self.agent_policy.get_action(obs, infos, deterministic=True)
                scenario_actions = self.scenario_policy.get_action(obs, infos, deterministic=True)

                # apply action to env and get obs
                next_obs, next_transition_obs, rewards, dones, next_infos, next_transition_infos = self.env.step(ego_actions, scenario_actions, onpolicy)

                infos = next_transition_infos
                obs = next_transition_obs

                # save video
                if self.save_video:
                    self.logger.add_frame(pygame.surfarray.array3d(self.display).transpose(1, 0, 2))

                # accumulate scores of corresponding scenario
                reward_idx = 0
                for s_i in infos:
                    score = rewards[reward_idx]
                    score_list[s_i['scenario_id']].append(score)
                    reward_idx += 1

            # clean up all things
            self.logger.log(">> All scenarios are completed. Cleaning up all actors")
            self.env.clean_up()

            # save video
            if self.save_video:
                data_ids = [config.data_id for config in sampled_scenario_configs]
                self.logger.save_video(data_ids=data_ids, scenario_id=self.scenario_id, map_name=self.current_map)

            # print score for ranking
            self.logger.log(f'[{num_finished_scenario}/{data_loader.num_total_scenario}] Ranking scores (rewards) for batch scenario:', 'yellow')
            for s_i in score_list.keys():
                self.logger.log('\t Env id ' + str(s_i) + ': ' + str(np.mean(score_list[s_i])), 'yellow')

            # calculate evaluation results
            if self.eval_mode == 'analysis':
                score_function = get_route_scores
                all_running_results = self.logger.add_eval_results(map_name=self.current_map, records=self.env.running_results)  # running results is growing as the evaluation goes
                all_scores = score_function(all_running_results)  # the current statistical scores from the start to the current evaluation scenario
                self.logger.add_eval_results(map_name=self.current_map, scores=all_scores)
                self.logger.print_eval_results(map_name=self.current_map)  # the finial eval results represent the statistical score during the whole process of evaluation
                if len(self.env.running_results) % self.save_freq == 0 or len(data_loader) == 0:
                    self.logger.save_eval_results(map_name=self.current_map)

    def collect_feasibility_data(self, data_loader, file_name):
        # general buffer for both agent and scenario
        buffer, onpolicy = self.check_onpolicy()

        data_loader.set_mode("train")

        while not all(buffer.feasibility_full):
            # sample scenarios in this town (one town could contain multiple scenarios)
            # simulate multiple scenarios in parallel (usually 2 scenarios)
            sampled_scenario_configs, config_lengths = data_loader.sampler()
            # if sampled scenario config length < num_scenarios, means need to balance to storing placement within buffer
            buffer.check_scenario_id_for_saving(config_lengths)
            # reset the index counter to create endless loader
            data_loader.reset_idx_counter()

            obs, infos = self.env.reset(sampled_scenario_configs)

            # get ego vehicle from scenario
            self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos)

            # start loop
            while not self.env.all_scenario_done():
                # get action from agent policy and scenario policy (assume using one batch)
                ego_actions = self.agent_policy.get_action(obs, infos, deterministic=False)
                scenario_actions = self.scenario_policy.get_action(obs, infos, deterministic=False)

                # apply action to env and get obs
                next_obs, next_transition_obs, rewards, dones, next_infos, next_transition_infos = self.env.step(ego_actions, scenario_actions, onpolicy)
                # store to the replay buffer
                buffer.store([ego_actions, scenario_actions, obs, next_obs, rewards, dones], additional_dict=[infos, next_infos])
                # for transition
                infos = next_transition_infos
                obs = copy.deepcopy(next_transition_obs)

            self.logger.log(f'>> dataset length: {buffer.buffer_len}')

            self.logger.log('>> Start Cleaning', 'yellow')
            # end up environment
            self.env.clean_up()

        # save the feasibility data
        buffer.save_feasibility_data(file_name)
        self.logger.log(">> Successfully saved the offline data", 'yellow')
        self.logger.log('>> ' + '-' * 40)

    def run(self):
        # get scenario data of different maps, and cluster config according to the town
        config_by_map = scenario_parse(self.scenario_config, self.logger)
        for m_i in config_by_map.keys():  # for each town, the same town could include different scenario templates
            # initialize map and render
            self.current_map = m_i  # record the current running map name
            self._init_world(m_i)
            if self.eval_mode == 'render':
                self._init_renderer()

            # create scenarios within the vectorized wrapper
            self.env = VectorWrapper(
                self.env_params,
                self.scenario_config,
                self.world,
                self.birdeye_render,
                self.display,
                self.use_feasibility,
                self.agent_state_encoder,
                self.logger,
            )
            self.logger.log(">> Finish scenario initialization")

            # prepare data loader and buffer
            data_loader = ScenarioDataLoader(config_by_map[m_i], self.num_scenario, m_i, self.world)
            self.logger.log(">> Finish data loader preparation")

            # run with different modes
            if self.mode == 'eval':
                # create the eval dir for each town
                self.agent_policy.load_model(map_name=self.current_map)
                self.scenario_policy.load_model(map_name=self.current_map)
                self.agent_policy.set_mode('eval')
                self.scenario_policy.set_mode('eval')
                if self.use_feasibility:
                    # loading the feasibility policy model
                    self.feasibility_policy.load_model(map_name=self.current_map)
                    self.feasibility_policy.set_mode('eval')
                    assert self.feasibility_policy.continue_episode != 0, 'The scenario policy need well-trained feasibility network'
                if self.agent_state_encoder:
                    self.agent_state_encoder.load_ckpt()
                self.eval(data_loader)
            elif self.mode == 'train_agent':
                start_episode = self.check_continue_training(self.agent_policy)
                self.scenario_policy.load_model(map_name=self.current_map)
                self.agent_policy.set_mode('train')
                self.scenario_policy.set_mode('eval')
                if self.agent_state_encoder:
                    self.agent_state_encoder.load_ckpt()
                self.train(data_loader, start_episode)
            elif self.mode == 'train_scenario':
                start_episode = self.check_continue_training(self.scenario_policy)
                self.agent_policy.load_model(map_name=self.current_map)
                self.agent_policy.set_mode('eval')
                self.scenario_policy.set_mode('train')
                if self.use_feasibility:
                    # loading the feasibility policy model
                    self.feasibility_policy.load_model(map_name=self.current_map)
                    self.feasibility_policy.set_mode('eval')
                    # pass the feasibility net to the scenario policy
                    self.scenario_policy.set_feasibility_policy(self.feasibility_policy)
                    assert self.feasibility_policy.continue_episode != 0, 'The scenario policy need well-trained feasibility network'
                if self.agent_state_encoder:
                    self.agent_state_encoder.load_ckpt()
                self.train(data_loader, start_episode)
            elif self.mode == 'collect_feasibility_data':
                self.agent_policy.load_model(map_name=self.current_map)
                self.agent_policy.set_mode('eval')
                self.scenario_policy.load_model(map_name=self.current_map)
                self.scenario_policy.set_mode('eval')
                if self.agent_state_encoder:
                    self.agent_state_encoder.load_ckpt()
                exist, file_name = self.check_feasibility_data_exists()
                if not exist:
                    self.collect_feasibility_data(data_loader, file_name)
            else:
                raise NotImplementedError(f"Unsupported mode: {self.mode}.")

    def check_onpolicy(self, start_episode=None):
        onpolicy_agent = True if self.agent_policy.type == 'onpolicy' else False
        onpolicy_scenario = True if self.scenario_policy.type == 'onpolicy' else False
        onpolicy = {'agent': onpolicy_agent, 'scenario': onpolicy_scenario}
        buffer = None
        if self.mode == 'train_agent':
            if self.agent_policy.type == 'onpolicy':
                buffer = RolloutBuffer(
                    self.num_scenario, self.mode, self.agent_config, self.scenario_config,
                    self.feasibility_config, self.buffer_capacity, self.logger
                )
            else:
                buffer = ReplayBuffer(
                    self.num_scenario, self.mode, start_episode, self.scenario_policy.type, self.current_map,
                    self.agent_config, self.scenario_config, self.feasibility_config, self.buffer_capacity, self.logger
                )
        elif self.mode == 'train_scenario':
            if self.scenario_policy.type == 'onpolicy':
                buffer = RolloutBuffer(
                    self.num_scenario, self.mode, self.agent_config, self.scenario_config,
                    self.feasibility_config, self.buffer_capacity, self.logger
                )
            else:
                buffer = ReplayBuffer(
                    self.num_scenario, self.mode, start_episode, self.scenario_policy.type, self.current_map,
                    self.agent_config, self.scenario_config, self.feasibility_config, self.buffer_capacity, self.logger
                )
        elif self.mode == 'collect_feasibility_data':
            buffer = RolloutBuffer(
                self.num_scenario, self.mode, self.agent_config, self.scenario_config,
                self.feasibility_config, self.buffer_capacity, self.logger
            )

        return buffer, onpolicy

    def check_continue_training(self, policy):
        # load previous checkpoint
        policy.load_model(map_name=self.current_map)
        if policy.continue_episode == 0:
            start_episode = 1
            CarlaDataProvider.set_random_seed(start_episode)
            self.logger.log('>> Previous checkpoint not found. Training from scratch.')
        else:
            start_episode = policy.continue_episode + 1
            CarlaDataProvider.set_random_seed(start_episode)
            self.logger.log('>> Continue training from previous checkpoint.')
        return start_episode

    def check_feasibility_data_exists(self):
        scenario_name = "all" if self.scenario_id is None else 'Scenario' + str(self.scenario_id)
        file_path = os.path.join(self.feasibility_data_path, scenario_name + "_" + self.current_map)
        file_name = os.path.join(file_path, f"{self.agent_policy_type}_{self.scenario_policy_type}_data.hdf5")
        if os.path.isfile(file_name):
            self.logger.log(f'>> exist data on {file_path}', color='red')
            exist = True
        else:
            os.makedirs(file_path, exist_ok=True)
            exist = False
        return exist, file_name

    def close(self):
        pygame.quit()
        if self.env:
            self.env.clean_up()
