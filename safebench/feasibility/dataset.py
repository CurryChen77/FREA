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
import torch
from safebench.util.torch_util import CUDA


class OffRLDataset(object):
    def __init__(self, data_location=None, device=None):

        dataset_dict, self.dataset_len = self.get_dataset(h5path=data_location)
        self.dataset_dict = self.put_dataset_on_cuda(dataset_dict) if device == 'cuda:0' else dataset_dict

    @staticmethod
    def put_dataset_on_cuda(dataset_dict):
        # use all the keys in the self.dataset_dict
        keys = list(dataset_dict.keys())

        # transfer the dataset input torch tensor, and put them on the GPU or CPU devices
        torch_dataset_dict = {k: torch.tensor(dataset_dict[k]) for k in keys}
        torch_dataset_dict = {k: CUDA(v) for k, v in torch_dataset_dict.items()}

        return torch_dataset_dict

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
                'ego_min_dis': np.zeros(dataset_len, dtype=np.float32),
                'dones': np.zeros(dataset_len, dtype=np.float32)
            }
            # for all the group files
            for group_name, group in file.items():
                group_data_len = group.attrs['length']
                for name, data in group.items():
                    data_dict[name][index:index+group_data_len] = data
                index += group_data_len

        return data_dict, dataset_len

    def sample(self, batch_size):

        # generate the random sampling index
        index = CUDA(torch.randint(0, self.dataset_len, (batch_size,)))

        # use teh index select function to get the data at the specified dim and index
        sample = {k: v.index_select(dim=0, index=index) for k, v in self.dataset_dict.items()}

        return sample


if __name__ == '__main__':

    output_filename = 'merged_data.hdf5'

    base_directory = './data'

    for folder in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder)
        output_path = os.path.join(folder_path, output_filename)

        if os.path.isdir(folder_path):
            print(f"\nProcessing files in folder: {folder}")

            dataset = OffRLDataset(output_path)
            samples = dataset.sample(batch_size=4)
            print("sample shape:", samples)

