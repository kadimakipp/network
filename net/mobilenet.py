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

class MobileNet(nn.Module):
    def __init__(self, img_shape, num_classes):
        super(MobileNet, self).__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes


def main():
    pass


if __name__ == "__main__":
    import fire

    fire.Fire(main)