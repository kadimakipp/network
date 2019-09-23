#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: mobilenet.py
@time: 2019/9/23 下午5:23
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1,bias=False)
        self.bn = nn.BatchNorm2d(32)

    def forward(self, x):
        out = F.relu6(self.bn(self.conv(x)))
        return out

class Tail(nn.Module):
    def __init__(self, num_classes):
        super(Tail, self).__init__()
        self.conv_1280 = nn.Conv2d(320, 1280,kernel_size=1, stride=1, padding=0,bias=False)
        self.bn_1280 = nn.BatchNorm2d(1280)
        self.conv_end = nn.Conv2d(1280, num_classes,kernel_size=1,bias=False)

    def forward(self, x):
        out = F.relu6(self.bn_1280(self.conv_1280(x)))
        out = F.avg_pool2d(out, kernel_size=7)
        out = self.conv_end(out)
        return out



class MobileNetV2(nn.Module):
    def __init__(self, in_planes, expansion, out_planes, repeat_times,stride):
        super(MobileNetV2, self).__init__()
        inner_channels = in_planes*expansion
        # 1*1-conv2d
        self.conv1 = nn.Conv2d(in_planes,inner_channels,kernel_size=1,
                               stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(inner_channels)

        #dwise
        self.conv2_stride = nn.Conv2d(inner_channels,inner_channels,kernel_size=3,stride=stride,
                                      padding=1,groups=inner_channels,bias=False)
        self.conv2 = nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=1,
                                      padding=1,groups=inner_channels, bias=False)
        #linear-1*1-conv2d
        self.conv3 =  nn.Conv2d(inner_channels, out_planes, kernel_size=1,
                                stride=1, groups=1, bias=False)

        #当某个bottleneck重复出现时，'1*1-conv2d'的输入输出的通道数发生变化，不能再使用conv1了
        self.conv_inner = nn.Conv2d(out_planes,expansion*out_planes,
                                    kernel_size=1,bias=False)
        # 当某个bottleneck重复出现时，dwise的输入输出的通道数发生变化，不能再使用conv2_stride和conv2了
        self.conv_inner_with_stride = nn.Conv2d(expansion * out_planes, expansion * out_planes, kernel_size=3,
                                                stride=stride, padding=1, groups=out_planes,
                                                bias=False)  # layer==1 stride=s
        self.conv_inner_no_stride = nn.Conv2d(expansion * out_planes, expansion * out_planes, kernel_size=3, stride=1,
                                              padding=1, groups=out_planes, bias=False)  # layer>1  stride=1
        #当某个bottleneck重复出现时，'linear-1*1-conv2d'的输入输出的通道数发生变化，不能再使用了
        self.conv3_inner = nn.Conv2d(expansion*out_planes, out_planes, kernel_size=1, stride=1, groups=1, bias=False)
        #当某个bottleneck重复出现时，batchnorm的通道数也同样发生了变化
        self.bn_inner = nn.BatchNorm2d(expansion*out_planes)
        self.bn2   = nn.BatchNorm2d(out_planes)
        self.n = repeat_times

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn1(self.conv2_stride(out)))
        out = self.conv3(out)
        out = self.bn2(out)
        count = 2
        while (count <= self.n):
            temp = out
            out = F.relu6(self.bn_inner(self.conv_inner(out)))
            out = F.relu6(self.bn_inner(self.conv_inner_no_stride(out)))
            out = self.conv3_inner(out)
            out = self.bn2(out)
            out = out + temp
            count = count + 1
        return out

class Net(nn.Module):
    # [input_channels, t, c, n, s] 论文中的参数列表
    param = [[32, 1, 16, 1, 1],
             [16, 6, 24, 2, 2],
             [24, 6, 32, 3, 2],
             [32, 6, 64, 4, 2],
             [64, 6, 96, 3, 1],
             [96, 6, 160, 3, 2],
             [160, 6, 320, 1, 1]]

    def __init__(self, num_class):
        super(Net,self).__init__()
        self.layers = self._make_layers(num_class)

    def _make_layers(self,num_class):
        layer = []
        layer.append(Head())
        for i, pa in enumerate(self.param):
            layer.append(MobileNetV2(*pa))

        layer.append(Tail(num_class))
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.layers(x)
        return out

def main():
    net = Net(1000)
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.size())


if __name__ == "__main__":
    import fire

    fire.Fire(main)