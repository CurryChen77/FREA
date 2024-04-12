#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：train_feasibility.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/1/23
"""
import os.path as osp
from safebench.util.run_util import load_config
import torch
from safebench.util.torch_util import set_seed, set_torch_variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from safebench.feasibility.dataset import OffRLDataset
from safebench.feasibility import FEASIBILITY_LIST
from safebench.util.logger import Logger


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_route', type=str, default='safebench/feasibility/data')
    parser.add_argument('--map_list', '-maps', nargs='+', default=['Town05', 'Town02'])
    parser.add_argument('--data_filename', type=str, default='merged_data.hdf5')
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))))
    parser.add_argument('--feasibility_cfg', nargs='*', type=str, default='HJR.yaml')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    args_dict = vars(args)
    # set global parameters
    set_torch_variable(args.device)
    torch.set_num_threads(args.threads)
    seed = args.seed
    set_seed(seed)
    map_list = args.map_list

    # load feasibility config
    feasibility_config_path = osp.join(args.ROOT_DIR, 'safebench/feasibility/config', args.feasibility_cfg)
    feasibility_config = load_config(feasibility_config_path)
    feasibility_config.update(args_dict)
    min_dis_threshold = feasibility_config['min_dis_threshold']
    scenario_name = "all" if feasibility_config['scenario_id'] is None else 'Scenario' + str(feasibility_config['scenario_id'])
    # train multiple scenario maps
    for map_name in map_list:
        scenario_map_name = scenario_name + '_' + map_name

        # the route of the data need to be processed
        data_file_path = osp.join(args.ROOT_DIR, args.data_route, scenario_map_name, args.data_filename)
        assert osp.exists(data_file_path), f"file {args.data_filename} don't exists"

        # set the logger
        log_path = osp.join(args.ROOT_DIR, 'safebench/feasibility/train_log', 'min_dis_threshold_' + str(min_dis_threshold) + '_seed' + str(args.seed), scenario_map_name)
        log_exp_name = scenario_map_name
        logger = Logger(log_path, log_exp_name)

        # read config parameters
        train_episode = feasibility_config['train_episode']
        batch_size = feasibility_config['batch_size']
        save_freq = feasibility_config['save_freq']
        assert train_episode > save_freq, "train episode should be greater than save frequency"

        # init the writer
        writer = SummaryWriter(log_dir=log_path)

        # init the feasibility policy
        feasibility_policy = FEASIBILITY_LIST[feasibility_config['type']](feasibility_config, logger=logger)
        feasibility_policy.load_model(map_name)
        if feasibility_policy.continue_episode == 0:
            start_episode = 0
            logger.log('>> Previous checkpoint not found. Training from scratch.')
        else:
            start_episode = feasibility_policy.continue_episode
            logger.log('>> Continue training from previous checkpoint.')

        logger.log('>> ' + '-' * 40)
        logger.log('>> Feasibility Policy: ' + feasibility_config['type'], color="yellow")
        logger.log('>> Scenario and map: ' + scenario_map_name, color="yellow")

        # init the offline RL dataset
        dataset = OffRLDataset(data_file_path, device=args.device)

        for e_i in tqdm(range(start_episode, train_episode + 1)):
            feasibility_policy.train(dataset, writer, e_i)

            # save the model
            if e_i != start_episode and e_i % save_freq == 0:
                feasibility_policy.save_model(e_i, map_name)

        # close the tensorboard writer
        writer.close()
        logger.log(f'>> Finish training feasibility of {map_name}')
        logger.log('>> ' + '-' * 40)

