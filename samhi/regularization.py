#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: regularization.py
@time: 2020/3/27 下午3:41
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
pytorch 实现的正则化是隐式实现的，并且正则化只有l2正则，所以实现一个l1，l2正则。
https://blog.csdn.net/guyuealian/article/details/88426648#1.torch.optim%E4%BC%98%E5%8C%96%E5%99%A8%E5%AE%9E%E7%8E%B0L2%E6%AD%A3%E5%88%99%E5%8C%96
"""
class Regularization(nn.Module):
    def __init__(self, weight_decay, p=2):
        super(Regularization, self).__init__()
        """
        @net: network
        @weight_decay: weight decay param
        @p: l2 regularization p=2, l1 regularization p=1 
        """
        self.weight_decay = weight_decay
        self.p = p

    def to(self, device):
        self.device = device
        super().to(self.device)#调用超类
        return self

    def __get_weights(self, net):
        weights = []
        for name, weight in net.named_parameters():
            if "weight" in name:
                weights.append(weight)
        return weights

    def _regularization(self, weights):
        loss = 0
        for weight in weights:
            loss+=torch.norm(weight, self.p)

        return self.weight_decay*loss

    def forward(self, net):
        regularization_weight = self.__get_weights(net)
        return self._regularization(regularization_weight)

####
#惩罚不同的参数
####

class RegularizationNew(Regularization):
    def __init__(self, weight_decay, p):
        super(RegularizationNew, self).__init__(weight_decay, p)

    def __get_weights(self, net):
        weights = []
        for name, weight in net.named_parameters():
            if "bias" in name:
                weights.append(weight)
        return weights

    def forward(self, net):
        regularization_weight = self.__get_weights(net)
        return self._regularization(regularization_weight)

####
#how use it
####
import torchvision.models.mobilenet as mobile
def main():
    net = mobile.mobilenet_v2(pretrained=False).to("cuda")
    for name, weight in net.named_parameters():
        print(name, weight.shape, weight.device)

    reg_loss = Regularization(0.1, 2)
    ## 伪代码
    net_loss = torch.tensor([1.0])
    loss = net_loss+reg_loss(net)
    print(loss)


if __name__ == "__main__":
    import fire

    fire.Fire(main)