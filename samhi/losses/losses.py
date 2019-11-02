#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: losses.py
@time: 2019/11/2 下午5:33
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

def L1_fun():
    input = torch.ones(3,5, requires_grad=True,dtype=torch.float32)*2
    target = torch.ones(3,5,dtype=torch.float32)
    #becase backward this is a scale
    loss = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
    return loss(input, target)

def MSE_fun():
    input = torch.ones(3, 5, requires_grad=True, dtype=torch.float32) * 2
    target = torch.ones(3, 5, dtype=torch.float32)
    # becase backward this is a scale
    loss = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    return loss(input, target)

def CrossEntropyLoss():
    pass

def main(l='L1'):
    output = None
    if l in ['L1']:
        output = L1_fun()
    elif l in ['MSE']:
        output = MSE_fun()

    if output is not None:
        output.backward()
        output.detach_()
        print(output.numpy())






if __name__ == "__main__":
    import fire

    fire.Fire(main)