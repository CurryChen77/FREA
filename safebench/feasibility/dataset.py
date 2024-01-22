#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：dataset.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/1/22
"""
import numpy as np
import os
import h5py


class OffRLDataset(object):
    def __init__(self, data_location=None):
        self.dataset_dict, self.dataset_len = self.get_dataset(h5path=data_location)

    @staticmethod
    def get_dataset(h5path):

        with h5py.File(h5path, 'r') as file:
            dataset_len = file.attrs['length']
            action_dim = file.attrs['action_dim']
            obs_shape = file.attrs['obs_shape']

            index = 0
            data_dict = {
                'actions': np.zeros((dataset_len, action_dim), dtype=np.float32),
                'obs': np.zeros((dataset_len, *obs_shape), dtype=np.float32),
                'next_obs': np.zeros((dataset_len, *obs_shape), dtype=np.float32),
                'constraint_h': np.zeros(dataset_len, dtype=np.float32),
                'dones': np.zeros(dataset_len, dtype=np.float32)
            }
            # for all the group files
            for group_name, group in file.items():
                print(f"processing group {group_name}")
                group_data_len = group.attrs['length']
                for name, data in group.items():
                    data_dict[name][index:index+group_data_len] = data
                index += group_data_len
                print(f"finish loading group {group_name}")

        return data_dict, dataset_len


if __name__ == '__main__':

    output_filename = 'merged_data.hdf5'

    base_directory = './data'

    for folder in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder)
        output_path = os.path.join(folder_path, output_filename)

        if os.path.isdir(folder_path):
            print(f"\nProcessing files in folder: {folder}")

            offRLdataset = OffRLDataset(output_path)

