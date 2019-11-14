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
[reference](https://arxiv.org/pdf/1512.03385.pdf) ResNet
"""

class CNNInput(nn.Sequential):
    def __init__(self, inc, ouc, kernel=7):
        super(CNNInput, self).__init__(
            nn.Conv2d(in_channels=inc, out_channels=ouc, kernel_size=kernel, stride=2,padding=3),
            nn.BatchNorm2d(ouc),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )
"""
@BasicBlock
down sample
                input(64d)
                      |------------------------------ conv 1x1 stride=2 channel=128 
    conv 3x3 stride=2, channel=128  padding=1                           BN
                    BN+ReLU                                             |
     conv 3x3 stride=1, channel=128  padding=1                           |
                      BN                                                |
                    add ------------------------------------------------|
                      |
                    ReLU
residual
                input(64)
                      |-------------------------------------------------|
    conv 3x3 stride=1, channel=64  padding=1                            |
                    BN+ReLU                                             |
     conv 3x3 stride=1, channel=64  padding=1                           |
                      BN                                                |
                    add ------------------------------------------------|
                      |
                    ReLU
"""
class BasicRes(nn.Sequential):
    """
    Basic block
    """
    def __init__(self, inc):
        super(BasicRes, self).__init__(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inc),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inc)
        )
class BasicResD(nn.Sequential):
    """
    Basic block with down sample
    """
    def __init__(self,inc, ouc):
        super(BasicResD, self).__init__(
            nn.Conv2d(in_channels=inc,out_channels=ouc,kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ouc),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=ouc,out_channels=ouc,kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(ouc)
        )
class DownSample(nn.Sequential):
    """
    use input down sample
    """
    def __init__(self, inc, ouc):
        super(DownSample, self).__init__(
            nn.Conv2d(in_channels=inc, out_channels=ouc, kernel_size=1,stride=2),
            nn.BatchNorm2d(ouc)
        )

class Sample(nn.Module):
    """
    identity
    """
    def __init__(self):
        super(Sample, self).__init__()

    def forward(self,x):
        return x

class BasicBlock(nn.Module):
    def __init__(self,inc, ouc):
        super(BasicBlock, self).__init__()
        self.conv = BasicRes(inc)
        self.barnch = Sample()
        if inc!=ouc:
            self.conv = BasicResD(inc, ouc)
            self.barnch = DownSample(inc,ouc)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = self.barnch(x)
        out = self.conv(x)
        out += identity
        out = self.relu(out)

        return out

"""
@Bottleneck
down sample
                input(256d)
                      |------------------------------ conv 1x1 stride=2 channel=512
    conv 1x1 stride=2, channel=128                                     BN
                    BN+ReLU                                             |
    conv 3x3 stride=1, channel=128  padding=1                           |
                    BN+ReLU                                             |
    conv 1x1 stride=1, channel=512                                      |
                     BN                                                 |
                    add ------------------------------------------------|
                      |
                    ReLU
residual
                input(256d)
                      |-------------------------------------------------|
    conv 1x1 stride=1, channel=64                                       |
                    BN+ReLU                                             |
    conv 3x3 stride=1, channel=64  padding=1                            |
                    BN+ReLU                                             |
    conv 1x1 stride=1, channel=256                                      |
                     BN                                                 |
                    add-------------------------------------------------|
                      |
                    ReLU
"""

""" multi layer for input layer and res block link"""
class BottleneckResS(nn.Sequential):
    def __init__(self,inc,ouc):
        assert ouc//4==inc
        super(BottleneckResS, self).__init__(
            nn.Conv2d(in_channels=inc, out_channels=inc,kernel_size=1),
            nn.BatchNorm2d(inc),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inc,out_channels=inc, kernel_size=3, padding=1),
            nn.BatchNorm2d(inc),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inc, out_channels=ouc,kernel_size=1),
            nn.BatchNorm2d(ouc)
        )

class SampleS(nn.Sequential):
    def __init__(self,inc, ouc):
        assert ouc//4==inc
        super(SampleS, self).__init__(
            nn.Conv2d(in_channels=inc, out_channels=ouc,kernel_size=1),
            nn.BatchNorm2d(ouc)
        )


class BottleneckRes(nn.Sequential):
    def __init__(self,inc):
        mic = inc//4
        super(BottleneckRes, self).__init__(
            nn.Conv2d(in_channels=inc, out_channels=mic,kernel_size=1),
            nn.BatchNorm2d(mic),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mic,out_channels=mic, kernel_size=3, padding=1),
            nn.BatchNorm2d(mic),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mic, out_channels=inc,kernel_size=1),
            nn.BatchNorm2d(inc)
        )

class BottleneckResD(nn.Sequential):
    def __init__(self,inc,ouc):
        mic = inc//2
        assert ouc//2 == inc
        super(BottleneckResD, self).__init__(
            nn.Conv2d(in_channels=inc,out_channels=mic,kernel_size=1,stride=2),
            nn.BatchNorm2d(mic),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mic,out_channels=mic, kernel_size=3,padding=1),
            nn.BatchNorm2d(mic),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mic,out_channels=ouc,kernel_size=1),
            nn.BatchNorm2d(ouc),
        )

class BottleneckBlock(nn.Module):
    def __init__(self, inc, ouc):
        super(BottleneckBlock, self).__init__()
        self.conv = BottleneckRes(inc)
        self.branch = Sample()
        if inc!=ouc and ouc//2==inc:
            self.conv = BottleneckResD(inc, ouc)
            self.branch = DownSample(inc,ouc)
        elif inc!=ouc and ouc//4 ==inc:
            self.conv = BottleneckResS(inc, ouc) #for input-layer and res body link
            self.branch = SampleS(inc, ouc)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.branch(x)
        out = self.conv(x)
        out += identity
        out = self.relu(out)
        return out

class CNNOutput(nn.Module):
    def __init__(self, inc, n_classes):
        super(CNNOutput, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inc, n_classes)

    def forward(self,x):
        out = self.pool(x)
        out.squeeze_()
        out = self.fc(out)
        return out
#For sample only implementation 18-layer and 50-layer
class ResNet18(nn.Module):
    def __init__(self,im_channel=3, n_classes=1000):
        super(ResNet18, self).__init__()
        self.head = CNNInput(im_channel,64)
        self.stage_one = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.stage_two = nn.Sequential(
            BasicBlock(64,128),
            BasicBlock(128, 128),
        )
        self.stage_three = nn.Sequential(
            BasicBlock(128, 256),
            BasicBlock(256, 256),
        )
        self.stage_four = nn.Sequential(
            BasicBlock(256, 512),
            BasicBlock(512, 512),
        )
        self.tail = CNNOutput(512, n_classes)

    def forward(self, x):
        out = self.head(x)
        out = self.stage_one(out)
        out = self.stage_two(out)
        out = self.stage_three(out)
        out = self.stage_four(out)
        out = self.tail(out)
        return out

class ResNet50(nn.Module):
    def __init__(self, im_channel=3, n_classes=1000):
        super(ResNet50, self).__init__()
        self.head = CNNInput(im_channel,64)
        self.stage_one = nn.Sequential(
            BottleneckBlock(64,  256),
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
        self.tail = CNNOutput(2048, n_classes)

    def forward(self, x):
        out = self.head(x)
        out = self.stage_one(out)
        out = self.stage_two(out)
        out = self.stage_three(out)
        out = self.stage_four(out)
        out = self.tail(out)
        return out

import unittest
class TestResNet(unittest.TestCase):
    def test_CNNInput(self):
        input = torch.randn(4, 3, 224, 224)
        cnn = CNNInput(3,32)
        out = cnn(input)
        self.assertEqual(out.shape, (4,32,56,56), "CNNInput Error")

    def test_CNNOutput(self):
        input = torch.randn(4,512,7,7)
        cnn = CNNOutput(512,1000)
        out = cnn(input)
        self.assertEqual(out.shape, (4,1000))

    def test_ResNet18(self):
        input = torch.randn(4, 3, 224, 224)
        resnet18 = ResNet18()
        out = resnet18(input)
        self.assertEqual(out.shape, (4, 1000))

    def test_ResNet50(self):
        input = torch.randn(4, 3, 224, 224)
        resnet50 = ResNet50()
        out = resnet50(input)
        self.assertEqual(out.shape, (4, 1000))


def main():
    unittest.main()

if __name__ == "__main__":
    import fire
    fire.Fire(main)