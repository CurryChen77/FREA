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

import numpy as np
from tqdm import tqdm


def merge_hdf5_files(output_filepath, input_filepaths):
    if not os.path.exists(output_filepath):
        with h5py.File(output_filepath, 'a') as output_file:
            buffer_len = 0
            for input_filepath in tqdm(input_filepaths, desc="Merging Files"):
                file_name = os.path.splitext(os.path.basename(input_filepath))[0]
                grp = output_file.create_group(file_name)
                with h5py.File(input_filepath, 'r') as input_file:
                    buffer_len += input_file.attrs['length']
                    for name, data in input_file.items():
                        grp.create_dataset(name, data=data)
                grp.attrs['length'] = buffer_len
    else:
        print("already exists merged data")


def find_h5_files(directory):
    h5_file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.h5')]
    return h5_file_paths


if __name__ == '__main__':

    # 指定输出文件名和输入文件列表
    output_filename = 'merged_data.h5'

    # 指定路径
    base_directory = './data'

    # 遍历每个文件夹
    for folder in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder)
        output_path = os.path.join(folder_path, output_filename)

        # 检查是否是文件夹
        if os.path.isdir(folder_path):
            print(f"\nProcessing files in folder: {folder}")

            # 获取文件夹中的所有 .h5 文件
            h5_files_paths = find_h5_files(folder_path)

            # 调用函数合并多个 HDF5 文件
            merge_hdf5_files(output_path, h5_files_paths)
