#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：setup.py
@Author  ：Keyu Chen
@mail    : chenkeyu7777@gmail.com
@Date    ：2023/10/4
@source  ：This project is modified from <https://github.com/trust-ai/SafeBench>
'''

from setuptools import setup, find_packages

setup(name='safebench',
      packages=["safebench"],
      include_package_data=True,
      version='1.0.0',
      install_requires=['gym', 'pygame'])
