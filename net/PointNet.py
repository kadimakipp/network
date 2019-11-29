#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: PointNet.py
@time: 2019/11/29 下午3:48
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

class ConvBasic(nn.Sequential):
    def __init__(self, inc, ouc, kernel=1):
        super(ConvBasic, self).__init__(
            nn.Conv1d(in_channels=inc, out_channels=ouc, kernel_size=kernel),
            nn.BatchNorm1d(ouc),
            nn.ReLU(inplace=True)
        )

class LinearBasic(nn.Sequential):
    def __init__(self, inf, ouf):
        super(LinearBasic, self).__init__(
            nn.Linear(in_features=inf, out_features=ouf),
            nn.BatchNorm1d(ouf),
            nn.ReLU(inplace=True)
        )

# T Net
class STNkd(nn.Module):
    def __init__(self, inc, k=3):
        super(STNkd, self).__init__()
        self.k = k
        self.up = nn.Sequential(
            ConvBasic(inc, 64),
            ConvBasic(64, 128),
            ConvBasic(128, 1024)
        )

        self.down = nn.Sequential(
            LinearBasic(1024, 512),
            LinearBasic(512, 256),
            nn.Linear(256, k ** 2)
        )

    def forward(self, x):
        """x:(batch_size, channel, num_points), channel=3 means xyz"""
        x = self.up(x) #[bn, 3, 1024]-->[bn, 1024, 1024]
        x = torch.max(x, dim=2, keepdim=True)[0].squeeze(dim=2) #[bn, 1024, 1024]-->[bn, 1024]
        x = self.down(x) # [bn, 1024]-->[bn,k**2]
        x = x.view(-1, self.k, self.k)
        iden = torch.eye(self.k,dtype=torch.float32,device=x.device)
        x = x+iden
        return x

class AlignmentNet(nn.Module):
    def __init__(self, inc, k=3):
        super(AlignmentNet, self).__init__()
        self.stn = STNkd(inc, k)

    def forward(self, x):
        t_net = self.stn(x) #[bn, k, k]
        x = x.transpose(2,1) #[bn, num_point, k]
        x = torch.bmm(x, t_net)
        x = x.transpose(2,1) #[bn, k, num_point]
        return x

class PointNetKernel(nn.Module):
    def __init__(self, inc):
        super(PointNetKernel, self).__init__()
        self.joint_align_3 = AlignmentNet(inc,3)
        self.joint_align_64 = AlignmentNet(64, 64)
        self.conv = ConvBasic(inc, 64)
        self.conv_end = nn.Sequential(
            ConvBasic(64,128),
            ConvBasic(128, 1024),
            nn.MaxPool1d(1024)
        )

    def forward(self, x):
        """
        :param x: [bn, 3, num_points]
        :return:
        """
        out = self.joint_align_3(x)
        out = self.conv(out)
        local_f = self.joint_align_64(out)
        out = self.conv_end(local_f)
        global_f = out.squeeze(dim=2)
        return local_f, global_f

def main():
    input = torch.randn(4,3, 1024)
    # # stn = STNkd(3)
    # align = AlignmentNet(3)
    # s = align(input)
    # print(s.shape)
    # input = torch.randn(4, 64, 1024)
    # stn = AlignmentNet(64, 64)
    # s = stn(input)
    # print(s.shape)

    pnet = PointNetKernel(3)
    loc, glo = pnet(input)
    print(loc.shape, glo.shape)



if __name__ == "__main__":
    import fire

    fire.Fire(main)