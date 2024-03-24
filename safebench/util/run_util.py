#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：run_util.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
"""

import os
import os.path as osp
import time
import numpy as np
from fnmatch import fnmatch

import yaml
import importlib

from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter


class VideoWriter:
    def __init__(self, filename='_autoplay.mp4', fps=10.0, **kw):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint64(img.clip(0, 1)*255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        # self.writer.write_frame(img)
        try:
            self.writer.write_frame(img)
        except:
            pass

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()


class VideoRecorder(object):
    def __init__(self, output_dir, logger):
        self.logger = logger
        self.output_dir = output_dir
        self.video_count = 0
        self.fps = 20
        self.frame_list = []

    def add_frame(self, frame):
        self.frame_list.append(frame)
    
    def save(self, data_ids, file_name):
        data_ids = ['{:04d}'.format(data_id) for data_id in data_ids]
        video_name = f'video_{"{:04d}".format(self.video_count)}_id_{"_".join(data_ids)}.mp4'
        file_path = os.path.join(self.output_dir, file_name, 'video')
        os.makedirs(file_path, exist_ok=True)
        video_file = os.path.join(file_path, video_name)
        self.logger.log(f'>> Saving video to {video_file}')

        # define video writer
        video_writer = VideoWriter(filename=video_file, fps=self.fps)
        for f in self.frame_list:
            video_writer.add(f)
        video_writer.close()

        # reset frame list
        self.frame_list = []
        self.video_count += 1


def print_dict(d):
    print(yaml.dump(d, sort_keys=False, default_flow_style=False))


def load_config(config_path="default_config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def find_config_dir(dir, depth=0):
    for path, subdirs, files in os.walk(dir):
        for name in files:
            if name == "config.yaml":
                return path, name
    # if we can not find the config file from the current dir, we search for the parent dir:
    if depth > 2:
        return None
    return find_config_dir(osp.dirname(dir), depth + 1)


def find_model_path(dir, itr=None):
    # if itr is specified, return model with the itr number
    if itr is not None:
        model_path = osp.join(dir, "model_" + str(itr) + ".pt")
        if not osp.exists(model_path):
            return None
            # raise ValueError("Model doesn't exist: " + model_path)
        return model_path
    # if itr is not specified, return model.pt or the one with the largest itr number
    pattern = "*pt"
    model = "model.pt"
    max_itr = -1
    for _, _, files in os.walk(dir):
        for name in files:
            if fnmatch(name, pattern):
                name = name.split(".pt")[0].split("_")
                if len(name) > 1:
                    itr = int(name[1])
                    if itr > max_itr:
                        max_itr = itr
                        model = "model_" + str(itr) + ".pt"
    model_path = osp.join(dir, model)
    if not osp.exists(model_path):
        return None
        # raise ValueError("Model doesn't exist: " + model_path)
    return model_path, max_itr


def setup_eval_configs(dir, itr=None):
    path, config_name = find_config_dir(dir)
    model_path, load_itr = find_model_path(osp.join(path, "model_save"), itr=itr)
    config_path = osp.join(path, config_name)
    configs = load_config(config_path)
    return model_path, load_itr, configs["policy"], configs["timeout_steps"], configs[configs["policy"]]


def class_from_path(path):
    module_name, class_name = path.rsplit(".", 1)
    class_object = getattr(importlib.import_module(module_name), class_name)
    return class_object
