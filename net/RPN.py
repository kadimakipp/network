#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: RPN.py
@time: 2019/11/26 上午11:45
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
[RPN](https://arxiv.org/pdf/1506.01497.pdf)
RPN is sample, but it need data is complex...
sliding window, 3x3 convolution...
"""
class RPN(nn.Module):
    pass


def main():
    pass


if __name__ == "__main__":
    import fire

    fire.Fire(main)