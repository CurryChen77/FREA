#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：eval_result.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/2/28
"""

import pickle
import os.path as osp


def read_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    ROOT_DIR = osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__))))
    record_path = osp.join(ROOT_DIR, 'log/eval/expert_ppo_rule-based_seed_0/eval_results/records.pkl')
    results_path = osp.join(ROOT_DIR, 'log/eval/expert_ppo_rule-based_seed_0/eval_results/results.pkl')
    record = read_pickle_file(record_path)
    print('>> ' + '-' * 40)

    results = read_pickle_file(results_path)
    print(results)


