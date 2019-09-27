#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: flops_test.py
@time: 19-5-31 上午11:00
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
import torch
import unittest
from model_tools_lib.flops import count_ops
from torchvision.models import resnet18


class Tester(unittest.TestCase):

    def test_overall(self):
        expected = 1826818048
        input = torch.rand(1, 3, 224, 224)
        net = resnet18()
        estimated = count_ops(net, input, print_readable=False)

        assert(expected == estimated)