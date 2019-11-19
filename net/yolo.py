#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: yolo.py
@time: 2019/11/19 下午7:10
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from net.darkent import DarkNet as backbone
"""
you only look once
V3
"""
# like darknet ResidualConv but some difference
class ConvOne(nn.Sequential):
    def __init__(self, inc, ouc):
        super(ConvOne, self).__init__(
            nn.Conv2d(in_channels=inc,out_channels=ouc,kernel_size=1),
            nn.BatchNorm2d(ouc),
            nn.LeakyReLU(inplace=True),
        )
class ConvThree(nn.Sequential):
    def __init__(self, inc,ouc):
        super(ConvThree, self).__init__(
            nn.Conv2d(in_channels=inc,out_channels=ouc,kernel_size=3,padding=1),
            nn.BatchNorm2d(ouc),
            nn.LeakyReLU(inplace=True)
        )

class ConvPred(nn.Module):
    def __init__(self, inc, ouc, out_size):
        super(ConvPred, self).__init__()
        





def main():
    pass


if __name__ == "__main__":
    import fire

    fire.Fire(main)