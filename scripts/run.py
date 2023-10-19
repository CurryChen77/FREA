#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：run.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
'''

import traceback
import os.path as osp

import torch 

from safebench.util.run_util import load_config
from safebench.util.torch_util import set_seed, set_torch_variable
from safebench.carla_runner import CarlaRunner


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--output_dir', type=str, default='log')
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__)))))

    parser.add_argument('--max_episode_step', type=int, default=300)
    parser.add_argument('--search_radius', type=int, default=60, help='the radius for agent to search other agents')
    parser.add_argument('--traffic_intensity', type=list, default=[0.5, 0.7, 0.6], help='the traffic intensity of the start, middle, end route positions')
    parser.add_argument('--auto_ego', action='store_true')
    parser.add_argument('--safety_eval', type=bool, default=True, help='whether to activate safety evaluation')
    parser.add_argument('--mode', '-m', type=str, default='eval', choices=['train_agent', 'train_scenario', 'eval', 'train_safety_network'])
    parser.add_argument('--agent_cfg', nargs='*', type=str, default='dummy.yaml')
    parser.add_argument('--scenario_cfg', nargs='*', type=str, default='standard.yaml')
    parser.add_argument('--safety_network_cfg', nargs='*', type=str, default='HJR.yaml')
    parser.add_argument('--continue_agent_training', '-cat', type=bool, default=False)
    parser.add_argument('--continue_scenario_training', '-cst', type=bool, default=False)

    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')   

    parser.add_argument('--num_scenario', '-ns', type=int, default=2, help='num of scenarios we run in one episode')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--render', type=bool, default=True)
    parser.add_argument('--frame_skip', '-fs', type=int, default=1, help='skip of frame in each step')
    parser.add_argument('--port', type=int, default=2000, help='port to communicate with carla')
    parser.add_argument('--tm_port', type=int, default=8000, help='traffic manager port')
    parser.add_argument('--fixed_delta_seconds', type=float, default=0.1)
    args = parser.parse_args()
    args_dict = vars(args)

    err_list = []
    for agent_cfg in args.agent_cfg:
        for scenario_cfg in args.scenario_cfg:
            # set global parameters
            set_torch_variable(args.device)
            torch.set_num_threads(args.threads)
            set_seed(args.seed)

            # load agent config
            agent_config_path = osp.join(args.ROOT_DIR, 'safebench/agent/config', agent_cfg)
            agent_config = load_config(agent_config_path)

            # load scenario config
            scenario_config_path = osp.join(args.ROOT_DIR, 'safebench/scenario/config', scenario_cfg)
            scenario_config = load_config(scenario_config_path)

            # if "train safety network mode" then must start safety eval
            safety_eval = True if args.mode == 'train_safety_network' else args.safety_eval
            if safety_eval:
                # load safety network config
                safety_network_config_path = osp.join(args.ROOT_DIR, 'safebench/safety_network/config', args.safety_network_cfg)
                safety_network_config = load_config(safety_network_config_path)
                safety_network_config.update(args_dict)
            else:
                safety_network_config = None

            # main entry with a selected mode
            agent_config.update(args_dict)
            scenario_config.update(args_dict)

            if scenario_config['policy_type'] == 'scenic':
                from safebench.scenic_runner import ScenicRunner
                assert scenario_config['num_scenario'] == 1, 'the num_scenario can only be one for scenic now'
                runner = ScenicRunner(agent_config, scenario_config)
            else:
                runner = CarlaRunner(agent_config, scenario_config, safety_network_config)  # create the main runner

            # start running
            try:
                runner.run()
            except:
                runner.close()
                traceback.print_exc()
                err_list.append([agent_cfg, scenario_cfg, traceback.format_exc()])

    for err in err_list:
        print(err[0], err[1], 'failed!')
        print(err[2])
