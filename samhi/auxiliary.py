#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: auxiliary.py
@time: 2019/9/18 下午8:00
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torchvision

class AuxFunction(object):
    def __init__(self):
        pass

    @staticmethod
    def device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def to_categorical(y, num_columns):
        """Return one-hot encoded device variable"""
        y_cat = torch.zeros((y.shape[0]), num_columns)
        y_cat[range(y.shape[0]), y] = 1.0

        return y_cat.to(device=AuxFunction.device())

    @staticmethod
    def update_lr(optimizer, lr):
        for params in optimizer.param_groups:
            params['lr'] = lr
