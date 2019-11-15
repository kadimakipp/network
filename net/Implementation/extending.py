#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: extending.py
@time: 2019/11/15 下午6:40
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

class LinearF(torch.autograd.Function):
    #Note that both forward and backward are @staticmethod
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        #These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias

#Now, to make it easier to use these custom ops, we recommend aliasing their apply method:
# `linear = LinearFunction.apply`

class MulConstant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, constant):
        #ctx is a context object that can be used to stash information
        #for backward computation
        ctx.constant = constant
        return tensor*constant

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments
        # Gradients of non-Tensor arguments to forward must be None
        return grad_output*ctx.constant, None

from torch.autograd import gradcheck
import unittest
class LinearTest(unittest.TestCase):
    def test_LinearTest(self):
        linear = LinearF.apply
        input = (torch.randn(20,20,dtype=torch.double,requires_grad=True),
                 torch.randn(30,20,dtype=torch.double,requires_grad=True))
        test = gradcheck(linear, input,
                         eps=1e-6, atol=1e-4)
        print(test)

        self.assertEqual(test, True)

    def test_MulConstant(self):
        mul = MulConstant.apply
        input = (torch.randn(2, dtype=torch.double, requires_grad=True),
                 3)
        test = gradcheck(mul, input)
        print(test)
        self.assertEqual(test, True)


def main():
    unittest.main()

if __name__ == "__main__":
    import fire

    fire.Fire(main)