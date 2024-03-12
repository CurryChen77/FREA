#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：test.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2024/3/12
"""


def calculate_collision_rate(done, collision):
    total_episodes = 0
    collision_episodes = 0

    has_collision = False

    for d, c in zip(done, collision):
        if d:  # episode 结束
            total_episodes += 1
            if has_collision or c:
                collision_episodes += 1
            has_collision = False
        else:
            has_collision = True if c else False

    collision_rate = collision_episodes / total_episodes if total_episodes != 0 else None
    return collision_rate

# 测试
done = [False, False, True, False, True, False, False, False, False, False, True]
coll = [False, True, False, False, False, False, False, True, True, False, True]

collision_rate = calculate_collision_rate(done, coll)
print("发生碰撞的比率:", collision_rate)


