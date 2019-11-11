#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: ResNet.py
@time: 2019/11/8 下午4:51
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

"""
[reference](https://zhuanlan.zhihu.com/p/54289848) and others
"""

class CNNInput(nn.Sequential):
    def __init__(self, inc, ouc, kernel):
        super(CNNInput, self).__init__(
            nn.Conv2d(in_channels=inc, out_channels=ouc, kernel_size=kernel),
            nn.BatchNorm2d(ouc),
            nn.ReLU(inplace=True),
        )

class ResNetBlock(nn.Sequential):
    def __init__(self, inc, ouc, kernel=3, stride=1):
        super(ResNetBlock, self).__init__(
            nn.Conv2d(in_channels=inc, out_channels=ouc, kernel_size=kernel, stride=stride, padding=1),
            nn.BatchNorm2d(ouc),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=ouc, out_channels=ouc, kernel_size=kernel, stride=stride, padding=1),
            nn.BatchNorm2d(ouc),
        )

def main():
    pass


if __name__ == "__main__":
    import fire

    fire.Fire(main)