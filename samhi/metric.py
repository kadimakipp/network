#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: metric.py
@time: 2019/11/11 下午3:16
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

class Metric(object):
    def __init__(self):
        pass

    @classmethod
    def error(cls, output, target):
        if output.is_cuda and target.is_cuda:
            out = output.cpu()
            tar = target.cpu()
        elif output.is_cuda and target.is_cuda is False:
            out = output.cpu()
            tar = target
        else:
            out = output
            tar = target
        _, pred = out.max(dim=1)
        return pred.eq(tar).sum()

    @classmethod
    def error_mask(cls, output, target, ignore=2):
        if output.is_cuda and target.is_cuda:
            out = output.cpu()
            tar = target.cpu()
        elif output.is_cuda and target.is_cuda is False:
            out = output.cpu()
            tar = target
        else:
            out = output
            tar = target
        mask = torch.ones_like(tar)
        mask[tar==ignore] = 0
        _,pred = out.max(dim=1)
        return pred[mask==1].eq(tar[mask==1]).sum(),mask.sum()

    @classmethod
    def PR(cls, output, target, threshold=0.5):
        """Compute total pixel-wise precision/recall,
           binary classifier
           threshold  max() - 0.01 99.99%
           precision = TP/(TP+FP)
           recall = TP/(TP+FN)
        """
        if output.is_cuda and target.is_cuda:
            out = output.cpu()
            tar = target.cpu()
        elif output.is_cuda and target.is_cuda is False:
            out = output.cpu()
            tar = target
        else:
            out = output
            tar = target
        out = out.squeeze(dim=1)
        TP = out[tar==1]>threshold
        FP = out[tar==0]>threshold
        FN = out[tar==1]<=threshold
        TN = out[tar==0]<=threshold
        TP = TP.sum().float()
        FP = FP.sum().float()
        FN = FN.sum().float()
        TN = TN.sum().float()
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        F1 = 2*P*R/(P+R)
        A = (TP+TN)/(TP+FP+FN+TN)
        return P, R, F1, A

    @classmethod
    def mAP(cls, output, target):
        """
        由PASCAL VOC challenge 提出来的计算方法，首先设定一组阈值[0,0.1,0.2,...,1.0],
        然后对于recall大于每一个阈值（比如recall>0.3），得到一个最大precision,最终得到11个precision，
        AP就是这11个precision的平均值，---11-point interpolated average precision。​
        2010年后提出新的方法，假设这N个样本中有M个正例，那么会得到M个recall值（1/M, 2/M, ..., M/M）,
        对于每个recall值r，可以计算出对应（r' > r）的最大precision，然后对这M个precision值取平均即得到最后的AP值。计算方法如下：​
        [reference1](https://medium.com/@hfdtsinghua/calculate-mean-average-precision-map-for-multi-label-classification-b082679d31be)
        [reference2](https://towardsdatascience.com/implementation-of-mean-average-precision-map-with-non-maximum-suppression-f9311eb92522)
        [reference3](https://github.com/Cartucho/mAP)
        """




def main():
    output = torch.empty(1,3,5,5).random_(2)
    target = torch.empty(1,5,5).random_(3)
    right = Metric.error(output, target)
    print(right.numpy())
    right, total = Metric.error_mask(output, target, ignore=2)
    print("Accuracy: ",right.numpy()/total.numpy()*100)
    output = torch.empty(1,1,5,5).random_(10).float()/10
    #output = torch.zeros_like(output)
    target = torch.empty(1,5,5).random_(2)
    #target = torch.ones_like(target)
    print(output, target)
    P, R, F1, A = Metric.PR(output, target)
    print(P, R, F1, A)




if __name__ == "__main__":
    import fire

    fire.Fire(main)