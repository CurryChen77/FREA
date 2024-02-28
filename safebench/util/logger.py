#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：logger.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""

import atexit
import json
import os
import os.path as osp
import time

import joblib
import numpy as np
import yaml

from safebench.util.run_util import VideoRecorder


# Where experiment outputs are saved by default:
DEFAULT_DATA_DIR = osp.abspath(osp.dirname(osp.dirname(osp.dirname(__file__))))

# Whether to automatically insert a date and time stamp into the names of
# save directories:
FORCE_DATESTAMP = False


def setup_logger_kwargs(exp_name, output_dir, seed, mode, datestamp=False, agent=None, agent_obs_type=None,
                        scenario=None, feasibility=None, scenario_id=None, CBV_selection=None):
    # Datestamp forcing
    datestamp = datestamp or FORCE_DATESTAMP

    # Make base path
    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
    relpath = ''.join([ymd_time, mode])
    # relpath = ''.join([ymd_time, exp_name])

    # specify agent policy and scenario policy in the experiment directory.
    agent_scenario_safety_net_exp_name = exp_name
    if agent is not None:
        agent_scenario_safety_net_exp_name = agent_scenario_safety_net_exp_name + '_' + agent + '(' + agent_obs_type + ')'
    if scenario is not None:
        agent_scenario_safety_net_exp_name = agent_scenario_safety_net_exp_name + '_' + scenario + '(' + CBV_selection + ')'
    if feasibility is not None:
        agent_scenario_safety_net_exp_name = agent_scenario_safety_net_exp_name + '_' + feasibility

    # Make a seed-specific subfolder in the experiment directory.
    if datestamp:
        hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        subfolder = ''.join([hms_time, '-', agent_scenario_safety_net_exp_name, '_s', str(seed)])
    else:
        subfolder = ''.join([agent_scenario_safety_net_exp_name, '_seed_', str(seed)])
    relpath = osp.join(relpath, subfolder)

    data_dir = os.path.join(DEFAULT_DATA_DIR, output_dir)
    logger_kwargs = dict(
        output_dir=osp.join(data_dir, relpath),
        exp_name=exp_name,
    )
    return logger_kwargs


def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False


def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}
        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)
        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]
        elif hasattr(obj, '__name__') and not ('lambda' in obj.__name__):
            return convert_json(obj.__name__)
        elif hasattr(obj, '__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v) for k, v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)


def statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.
    Args:
        x: An array containing samples of the scalar to produce statistics
            for.
        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    mean = np.mean(x)
    std = np.std(x)  # compute global std
    if with_min_and_max:
        return mean, std, np.min(x), np.max(x)
    return mean, std


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


class Logger:
    """
        A general-purpose logger.
        Makes it easy to save diagnostics, hyperparameter configurations, the state of a training run, and the trained model.
    """
    def __init__(self, output_dir=None, output_fname='log.txt', exp_name=None):
        """
            Initialize a Logger.

            Args:
                output_dir (string): A directory for saving results to. 
                    If ``None``, defaults to a temp directory of the form ``/tmp/experiments/somerandomnumber``.

                output_fname (string): Name for the tab-separated-value file 
                    containing metrics logged throughout a training run. Defaults to ``progress.txt``. 

                exp_name (string): Experiment name. If you run multiple training
                    runs and give them all the same ``exp_name``, the plotter will know to group them. (Use case: if you run the same
                    hyperparameter configuration with multiple random seeds, you should give them all the same ``exp_name``.)
        """
        self.epoch = 0
        self.exp_name = exp_name
        self.video_recorder = None

        self.output_dir = output_dir or "/tmp/experiments/%i" % int(time.time())
        os.makedirs(self.output_dir, exist_ok=True)
        # self.output_file = open(osp.join(self.output_dir, output_fname), 'a')
        # atexit.register(self.output_file.close)
        self.log('>> ' + '-' * 40)
        self.log(">> Logging to %s" % self.output_dir, 'green')
        self.log('>> ' + '-' * 40)
        
        self.eval_results = {}
        self.eval_records = {}

    def create_eval_dir(self, load_existing_results=True):
        result_dir = os.path.join(self.output_dir, 'eval_results')
        os.makedirs(result_dir, exist_ok=True)
        self.result_file = os.path.join(result_dir, 'results.pkl')
        self.record_file = os.path.join(result_dir, 'records.pkl')
        if load_existing_results:
            if os.path.exists(self.record_file):
                self.log(f'>> Loading existing evaluation records from {self.record_file}')
                self.eval_records = joblib.load(self.record_file)
            else:
                self.log(f'>> Loading existing record fail because no records.pkl is found.')
                self.eval_records = {}

    def add_eval_results(self, scores=None, records=None):
        if scores is not None:
            self.eval_results.update(scores)
        if records is not None:
            self.eval_records.update(records)
            return self.eval_records

    def save_eval_results(self):
        self.log(f'>> Saving evaluation results to {self.result_file}')
        joblib.dump(self.eval_results, self.result_file)
        self.log(f'>> Saving evaluation records to {self.record_file}, length: {len(self.eval_records)}')
        joblib.dump(self.eval_records, self.record_file)

    def print_eval_results(self):
        self.log("Evaluation results up to now:")
        for key, value in self.eval_results.items():
            self.log(f"\t {key: <25}{value}")

    def log(self, msg, color='green'):
        # print with color
        print(colorize(msg, color, bold=True))

    def log_dict(self, dict_msg, color='green'):
        for key, value in dict_msg.items():
            self.log("{}: {}".format(key, value), color)

    def save_config(self, config):
        """
            Log an experiment configuration.
        """
        if self.exp_name is not None:
            config['exp_name'] = self.exp_name
        config_json = convert_json(config)
        output = json.dumps(config_json, separators=(',', ':\t'), indent=4, sort_keys=True)
        with open(osp.join(self.output_dir, "config.json"), 'w') as out:
            out.write(output)
    
    def init_video_recorder(self):
        self.video_recorder = VideoRecorder(self.output_dir, logger=self)

    def add_frame(self, frame):
        self.video_recorder.add_frame(frame)

    def save_video(self, data_ids):
        self.video_recorder.save(data_ids=data_ids)
