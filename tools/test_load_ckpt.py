#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：test_load_ckpt.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/26
"""
import torch
from safebench.agent.agent_utils.agent_state_encoder import EncoderModel
from safebench.util.run_util import load_config
from pytorch_lightning.utilities.cloud_io import load as pl_load
import os.path as osp
import argparse
from collections import OrderedDict


def update_state_encoder(model_path, updated_model_path):
    checkpoint = pl_load(model_path, map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        print("old key:", key)
        cleaned_key = key[len("model."):] if key.startswith("model.") else key
        print("old key:", cleaned_key)
        print("_________________________")
        if cleaned_key.startswith("heads") or cleaned_key.startswith("wp") or cleaned_key == "model.embeddings.position_ids":
            continue
        else:
            new_state_dict[cleaned_key] = value
    checkpoint["state_dict"] = new_state_dict
    save_path = updated_model_path
    torch.save(checkpoint, save_path)
    print("successfully saving the updated state encoder ckpt")


def main(args):
    strict = True
    ROOT_DIR = osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__))))
    config_path = osp.join(ROOT_DIR, 'safebench/safety_network/config/HJR.yaml')
    config = load_config(config_path)
    model_path = osp.join(ROOT_DIR, 'safebench/agent/model_ckpt/PlanT_state_encoder/checkpoints/epoch=047.ckpt')
    updated_model_path = osp.join(ROOT_DIR, config['pretrained_model_path'])

    if args.Transfer_OriginCKPT_to_StateEncoderCKPT:
        update_state_encoder(model_path, updated_model_path)

    # load the already transferred state encoder ckpt
    checkpoint = pl_load(updated_model_path, map_location=lambda storage, loc: storage)
    model = EncoderModel(config)
    keys = model.load_state_dict(checkpoint["state_dict"], strict=strict)

    if not strict:
        if keys.missing_keys:
            print(
                f"missing keys: {keys.missing_keys}"
            )
        else:
            print("no missing keys")
        if keys.unexpected_keys:
            print(
                f"unexpected keys: {keys.unexpected_keys}"
            )
        else:
            print("no unexpected keys")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Transfer_OriginCKPT_to_StateEncoderCKPT', type=str, default=False)
    args = parser.parse_args()

    main(args)