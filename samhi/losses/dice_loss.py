#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: dice_loss.py
@time: 2019/12/13 上午11:48
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch.nn import _reduction as _Reduction

class DiceLoss(nn.Module):
    def __init__(self, weight=None,smooth=True, size_average=None, reduce=None,reduction='mean'):
        super(DiceLoss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
        if smooth:
            self.smooth = 1.
        else:
            self.smooth = 0.
        self.weight = weight

    def forward(self, input:torch.Tensor, target:torch.Tensor):
        """

        :param input: pred value
        :param target: one hot label
        :return:
        """
        dim = (1,)
        input = torch.sigmoid(input)
        print(input.shape, target.shape)
        intersection = (input*target).sum(dim=dim)
        X = input.sum(dim=dim)
        Y = target.sum(dim=dim)

        score = (2.*intersection+self.smooth)/(X+Y+self.smooth)

        if self.weight is not None:
            score *= self.weight
        if self.reduction != 'none':
            score = torch.mean(score) if self.reduction == 'mean' else torch.sum(score)

        loss = 1.-score

        return loss


def main():
    bdl = DiceLoss(smooth=True,reduction='mean')
    #2D
    input = torch.randn((4,1))#(4,2) one hot encode
    target = torch.randint(2,(4,1))
    print(input, target)
    loss = bdl(input, target)
    print(loss)
    #4D
    input = torch.randn((4,3,10,10))
    target = torch.randint(3,(4,10,10))
    one_hot = torch.nn.functional.one_hot(target, num_classes=3).unsqueeze(dim=1)
    one_hot.transpose_(1,-1).squeeze_(dim=-1)
    print(one_hot.shape)
    loss = bdl(input, one_hot)
    print(loss)

    _, index = torch.max(one_hot,dim=1)
    print(torch.equal(target, index))

if __name__ == "__main__":
    import fire

    fire.Fire(main)