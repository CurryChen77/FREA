#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：unify_offline_data.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/1/21
"""
import h5py
import os
import os.path as osp
from tqdm import tqdm
from typing import Dict


def merge_hdf5_files(output_filepath, input_filepaths):
    if not os.path.exists(output_filepath):
        with h5py.File(output_filepath, 'a') as output_file:
            total_len = 0
            action_dim = None
            obs_shape = None
            for input_filepath in tqdm(input_filepaths, desc="Merging Files"):
                file_name = os.path.splitext(os.path.basename(input_filepath))[0]
                grp = output_file.create_group(file_name)
                with h5py.File(input_filepath, 'r') as input_file:
                    total_len += input_file.attrs['length']
                    if action_dim is None:
                        action_dim = input_file.attrs['action_dim']
                    if obs_shape is None:
                        obs_shape = input_file.attrs['obs_shape']
                    for name, data in input_file.items():
                        # create the group data structure
                        grp.create_dataset(name, data=data)
                    grp.attrs['length'] = input_file.attrs['length']
            output_file.attrs['length'] = total_len
            output_file.attrs['action_dim'] = action_dim
            output_file.attrs['obs_shape'] = obs_shape
    else:
        print("already exists merged data")


def find_h5_files(directory):
    h5_file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.hdf5')]
    return h5_file_paths


if __name__ == '__main__':

    output_filename = 'merged_data.hdf5'
    ROOT_DIR = osp.abspath(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))))

    base_directory = osp.join(ROOT_DIR, 'frea/feasibility/data')

    for folder in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder)
        output_path = os.path.join(folder_path, output_filename)

        if os.path.isdir(folder_path):
            print(f"\nProcessing files in folder: {folder}")

            h5_files_paths = find_h5_files(folder_path)

            merge_hdf5_files(output_path, h5_files_paths)
