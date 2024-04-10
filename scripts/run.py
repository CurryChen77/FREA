#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：run.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""

import traceback
import os.path as osp

import torch 

from safebench.util.run_util import load_config
from safebench.util.torch_util import set_seed, set_torch_variable
from safebench.carla_runner import CarlaRunner


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='log')
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__)))))

    parser.add_argument('--CBV_selection', '-CBV', type=str, default='rule-based', choices=['rule-based', 'attention-based'])
    parser.add_argument('--auto_ego', action='store_true')
    parser.add_argument('--viz_route', '-vr', action='store_true')
    parser.add_argument('--use_feasibility', '-fe', type=bool, default=True)
    parser.add_argument('--mode', '-m', type=str, default='eval', choices=['train_agent', 'train_scenario', 'eval', 'collect_feasibility_data'])
    parser.add_argument('--eval_mode', '-em', type=str, default='store_data', choices=['store_data', 'render'])
    parser.add_argument('--agent_cfg', nargs='*', type=str, default='expert.yaml')
    parser.add_argument('--scenario_cfg', nargs='*', type=str, default='standard_eval.yaml')
    parser.add_argument('--feasibility_cfg', nargs='*', type=str, default='HJR.yaml')

    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')   

    parser.add_argument('--num_scenario', '-ns', type=int, default=2, help='num of scenarios we run in one episode')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--spectator', '-sp', action='store_true', default=False)
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

            # load feasibility config
            feasibility_config_path = osp.join(args.ROOT_DIR, 'safebench/feasibility/config', args.feasibility_cfg)
            feasibility_config = load_config(feasibility_config_path)
            feasibility_config.update(args_dict)

            # eval_mode only works when evaluating
            if args.mode != 'eval':
                args.eval_mode = None
                args.save_video = False

            # only save video when the eval mode is 'render'
            if args.eval_mode != 'render':
                args.save_video = False
            if args.eval_mode != 'store_data':
                args.use_feasibility = False

            # main entry with a selected mode
            agent_config.update(args_dict)
            scenario_config.update(args_dict)

            runner = CarlaRunner(agent_config, scenario_config, feasibility_config)  # create the main runner

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
