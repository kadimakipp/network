#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: darkent.py
@time: 2019/11/19 下午4:33
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
YOLOv3 https://pjreddie.com/media/files/papers/YOLOv3.pdf
https://blog.csdn.net/qq_37541097/article/details/81214953
"""

class ResidualConv(nn.Sequential):
    def __init__(self, inc,mic,ouc):
        assert inc==ouc
        super(ResidualConv, self).__init__(
            nn.Conv2d(in_channels=inc,out_channels=mic,kernel_size=1),
            nn.BatchNorm2d(mic),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=mic,out_channels=ouc,kernel_size=3,padding=1),
            nn.BatchNorm2d(ouc),
            nn.LeakyReLU(inplace=True)
        )

class Sample(nn.Module):
    def __init__(self):
        super(Sample, self).__init__()
        pass

    def forward(self, x):
        return x

class ConvDown(nn.Sequential):
    def __init__(self,inc,ouc):
        super(ConvDown, self).__init__(
            nn.Conv2d(in_channels=inc,out_channels=ouc,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(ouc),
            nn.LeakyReLU(inplace=True)
        )

class Residual(nn.Module):
    def __init__(self,inc, ouc, t):
        mic=ouc//2
        super(Residual, self).__init__()
        self.t = t
        self.conv_down = ConvDown(inc, ouc)
        self.res = ResidualConv(ouc,mic,ouc)
        self.sample=Sample()


    def forward(self,x):
        out = self.conv_down(x)
        for i in range(self.t):
            identity = self.sample(out)
            res = self.res(out)
            out = identity+res
        return out

class CNNInput(nn.Sequential):
    def __init__(self,inc,ouc):
        super(CNNInput, self).__init__(
            nn.Conv2d(in_channels=inc,out_channels=ouc,kernel_size=3,padding=1),
            nn.BatchNorm2d(ouc),
            nn.LeakyReLU(inplace=True)
        )

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

class DarkNet(nn.Module):
    def __init__(self, inc):
        super(DarkNet, self).__init__()
        self.cnn_input = CNNInput(inc=inc,ouc=32)
        self.stage_1 = Residual(32, 64, 1)
        self.stage_2 = Residual(64, 128, 2)
        self.stage_3 = Residual(128, 256, 8)
        self.stage_4 = Residual(256, 512, 8)
        self.stage_5 = Residual(512, 1024, 4)

    def forward(self, x):
        feature = self.cnn_input(x)
        feature = self.stage_1(feature)
        feature = self.stage_2(feature)
        one = self.stage_3(feature)
        two = self.stage_4(one)
        three = self.stage_5(two)
        out = {'one':one,'two':two,'three':three}

        return out

class DarkNet53(nn.Module):
    def __init__(self,inc, n_classes):
        super(DarkNet53, self).__init__()
        self.cnn_input = CNNInput(inc=inc,ouc=32)

        self.stage_1 = Residual(32,64,1)
        self.stage_2 = Residual(64,128,2)
        self.stage_3 = Residual(128,256,8)
        self.stage_4 = Residual(256,512,8)
        self.stage_5 = Residual(512,1024,4)

        self.cnn_out = CNNOutput(1024,n_classes)

    def forward(self,x):
        fea = self.cnn_input(x)
        one = self.stage_1(fea)
        two = self.stage_2(one)
        three = self.stage_3(two)
        four = self.stage_4(three)
        five = self.stage_5(four)

        out = self.cnn_out(five)
        o={'out':out,
           'one':one,
           'two':two,
           'three':three,
           'four':four,
           'five':five}
        return o

def main():
    inp = torch.randn([2,3,256,256])
    darknet = DarkNet53(3, 1000)
    out = darknet(inp)
    for o in out.keys():
        print(o, out[o].shape)
    inp = torch.randn([4,3,416,416])
    yolo_backbone = DarkNet(3)
    out = yolo_backbone(inp)
    for o in out.keys():
        print(o, out[o].shape)

if __name__ == "__main__":
    import fire

    fire.Fire(main)