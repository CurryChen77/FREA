#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：__init__.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
'''

# collect policy models from scenarios
from safebench.safety_network.HJ_Reachability import HJR


SAFETY_NETWORK_LIST = {
    'HJ-Reachability': HJR
}
