#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：test_env.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
'''

import gym
import gym_carla

from planning.env_wrapper import carla_env


def print_env_info(env):
    print(
        f"Observation space shape: {env.observation_space.shape}, \
        low: {env.observation_space.low}, \
        high: {env.observation_space.high}"
    )
    print(
        f"Action space shape: {env.action_space.shape}, \
        low: {env.action_space.low}, \
        high: {env.action_space.high}"
    )
    print("Env id: ", env.spec.id)
    print("Env max time steps: ", env.spec.max_episode_steps)


def runner(env_fn: gym.Env):
    # parameters for the gym_carla environment

    # Set gym-carla environment
    env = env_fn()
    env.seed(0)
    print_env_info(env)

    obs = env.reset()

    while True:
        action = [1.0, 0.0]
        obs, r, done, info = env.step(action)
        print(obs)
        if done:
            obs = env.reset()


if __name__ == '__main__':
    runner(env_fn=carla_env)
