#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: SENet.py
@time: 19-9-27 下午11:14
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

class SE(nn.Sequential):
    def __init__(self, inp, k):
        super(SE, self).__init__(
            nn.AvgPool2d(kernel_size=k),
            nn.Conv2d(inp, inp//8, kernel_size=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(inp//8, inp, kernel_size=1),
            nn.Sigmoid()
        )

#test
class S(nn.Module):
    def __init__(self, inp, k):
        super(S,self).__init__()
        self.se = SE(inp,k)

    def forward(self, x):
        return x*self.se(x)



if __name__ == "__main__":
    se = SE(16,7)
    print(se)
    features = torch.randn(1,16,7,7)
    out = se(features)
    # print(out.shape[1])
    # out = features*out
    print(out.shape)

