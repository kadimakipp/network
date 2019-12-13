#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: focal_loss.py
@time: 2019/12/12 上午9:13
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import _reduction as _Reduction
"https://arxiv.org/pdf/1708.02002.pdf"

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=None,size_average=None,ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        if gamma is None:
            gamma=1
        self.gamma=gamma
        self.size_average=size_average
        self.ignore_index=ignore_index
        self.reduce=reduce
        self.reduction=reduction


    def forward(self, input:torch.Tensor, target):
        reduction = self.reduction
        if self.size_average is not None or self.reduce is not None:
            reduction = _Reduction.legacy_get_string(self.size_average, self.reduce)
        pt = F.softmax(input, dim=1)
        log_pt = torch.log(pt)
        w = torch.pow((1-pt),self.gamma)
        f_l = F.nll_loss(w*log_pt, target, self.weight, None, self.ignore_index, None, reduction)

        return f_l




def main():
    i = np.array([[0.7, 0.2], [0.4, 1.2]])
    t = np.array([0, 1])
    weight = torch.from_numpy(np.array([0.6, 0.2])).float()
    input = torch.from_numpy(i).float()
    target = torch.from_numpy(t).type(torch.LongTensor)
    loss_f = nn.CrossEntropyLoss(weight=weight,reduction='none')
    output = loss_f(input, target)
    print(output)
    loss_FL=FocalLoss(reduction='none')
    output = loss_FL(input, target)
    print(output)
    # -------
    su = np.exp(i).sum(axis=1).reshape(2,1)
    sf = np.exp(i)/su
    print(su, sf)
    log_sf = np.log(sf)
    nsf = 1-sf
    nfl = -nsf*log_sf
    fl = nfl[range(2),t]
    print(nfl,fl)

if __name__ == "__main__":
    import fire

    fire.Fire(main)