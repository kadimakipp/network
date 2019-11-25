#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: FPN.py
@time: 2019/11/25 下午5:14
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


"""
[FPN](https://arxiv.org/pdf/1612.03144.pdf)
[ResNet50](../net/ResNet.py)
MayBe
[Spatial Aware Pooling](https://arxiv.org/pdf/1911.07217.pdf)
"""

import numpy as np
import torch
import torch.nn as nn

from net.ResNet import BottleneckBlock
class CNNInput(nn.Sequential):
    def __init__(self, inc, ouc, kernel=7):
        super(CNNInput, self).__init__(
            nn.Conv2d(in_channels=inc, out_channels=ouc, kernel_size=kernel, stride=2,padding=3),
            nn.BatchNorm2d(ouc),
            nn.ReLU(inplace=True),
        )
class ResNet50(nn.Module):
    def __init__(self, inc):
        super(ResNet50, self).__init__()
        self.head = CNNInput(inc, 64)
        self.stage_one = nn.Sequential(
            BottleneckBlock(64, 256),
            BottleneckBlock(256, 256),
            BottleneckBlock(256, 256),
        )
        self.stage_two = nn.Sequential(
            BottleneckBlock(256, 512),
            BottleneckBlock(512, 512),
            BottleneckBlock(512, 512),
            BottleneckBlock(512, 512),
        )
        self.stage_three = nn.Sequential(
            BottleneckBlock(512, 1024),
            BottleneckBlock(1024, 1024),
            BottleneckBlock(1024, 1024),
            BottleneckBlock(1024, 1024),
            BottleneckBlock(1024, 1024),
            BottleneckBlock(1024, 1024),
        )
        self.stage_four = nn.Sequential(
            BottleneckBlock(1024, 2048),
            BottleneckBlock(2048, 2048),
            BottleneckBlock(2048, 2048),
        )
    def forward(self, x):
        out = self.head(x)

        one = self.stage_one(out)
        two = self.stage_two(one)
        three = self.stage_three(two)
        four = self.stage_four(three)

        return {0:one,1:two,2:three,3:four}



class ConvFusionOne(nn.Sequential):
    def __init__(self,inc, ouc=256):
        super(ConvFusionOne, self).__init__(
            nn.Conv2d(in_channels=inc,out_channels=ouc,kernel_size=1,stride=1),
            nn.BatchNorm2d(ouc),
            nn.ReLU(inplace=True)
        )
class ConvFusionThree(nn.Sequential):
    def __init__(self,inc, ouc=256):
        super(ConvFusionThree, self).__init__(
            nn.Conv2d(in_channels=inc, out_channels=ouc,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ouc),
            nn.ReLU(inplace=True)
        )
class FPN(nn.Module):
    def __init__(self, inc):
        super(FPN, self).__init__()
        self.resnet50 = ResNet50(inc=inc)







def main():
    input = torch.randn(4,3,512,512)
    res50 = ResNet50(3)
    output = res50(input)
    for i in range(4):
        print(output[i].shape)


if __name__ == "__main__":
    import fire

    fire.Fire(main)