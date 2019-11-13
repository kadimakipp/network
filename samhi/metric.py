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
        mAP for object detection
        由PASCAL VOC challenge 提出来的计算方法，首先设定一组阈值[0,0.1,0.2,...,1.0],
        然后对于recall大于每一个阈值（比如recall>0.3），得到一个最大precision,最终得到11个precision，
        AP就是这11个precision的平均值，---11-point interpolated average precision。​
        2010年后提出新的方法，假设这N个样本中有M个正例，那么会得到M个recall值（1/M, 2/M, ..., M/M）,
        对于每个recall值r，可以计算出对应（r' > r）的最大precision，然后对这M个precision值取平均即得到最后的AP值。计算方法如下：​
        [reference1](https://medium.com/@hfdtsinghua/calculate-mean-average-precision-map-for-multi-label-classification-b082679d31be)
        [reference2](https://towardsdatascience.com/implementation-of-mean-average-precision-map-with-non-maximum-suppression-f9311eb92522)
        [reference3](https://github.com/Cartucho/mAP)
        """
        pass

    """
    [reference](https://arxiv.org/abs/1704.06857)
    Semantic Segmentation Metric
    共k+1类, background+k classes
    有一个混淆矩阵的概念用的比较好
    P_ij  i为GT j为pred 表示属于i类预测为j类。
    P_ii  TP(True positive)
    P_ij  FP(False positive)
    P_ji  FN(False negative)
    P_jj  TN(True negative)
    """
    @classmethod
    def PA_old(cls, output, target):
        """Pixel Accuracy: 预测正确的占所有像素的比值"""
        _, indices = output.max(dim=1)
        row = target.shape[1]
        col = target.shape[2]
        total = row*col
        TP = indices.eq(target).sum().numpy()
        return TP/total

    @classmethod
    def PA(cls, output, target, threshold):
        """ PA_old threshold edition"""
        out, indices = output.max(dim=1)
        mask = out.gt(threshold)# > gt < lt
        row = target.shape[1]
        col = target.shape[2]
        total = row * col
        TP = indices[mask].eq(target[mask]).sum().numpy()
        return TP/total

    @classmethod
    def MAP(cls, output, target, threshold):
        """Mean Pixel Accuracy:"""
        pass




"""
Semantic Segmentation Metric
https://blog.csdn.net/u014593748/article/details/71698246
"""






def main():
    #classification
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

    #Semantic Segmentation
    k = 3
    # output = torch.empty(1,k,5,5).random_(10).float()/10
    # target = torch.empty(1,5,5).random_(k) #generator data
    # print(output, '\n', target)
    output = torch.tensor([[[[0.9000, 0.5000, 0.1000, 0.0000, 0.2000],
              [0.9000, 0.4000, 0.4000, 0.1000, 0.9000],
              [0.0000, 0.7000, 0.9000, 0.4000, 0.0000],
              [0.2000, 0.8000, 0.2000, 0.6000, 0.6000],
              [0.1000, 0.4000, 0.6000, 0.3000, 0.9000]],

             [[0.1000, 0.5000, 0.2000, 0.4000, 0.7000],
              [0.1000, 0.6000, 0.4000, 0.5000, 0.9000],
              [0.2000, 0.6000, 0.5000, 0.9000, 0.6000],
              [0.4000, 0.7000, 0.2000, 0.3000, 0.7000],
              [0.5000, 0.3000, 0.6000, 0.4000, 0.1000]],

             [[0.2000, 0.3000, 0.8000, 0.1000, 0.6000],
              [0.1000, 0.9000, 0.1000, 0.1000, 0.7000],
              [0.5000, 0.3000, 0.6000, 0.1000, 0.8000],
              [0.2000, 0.6000, 0.1000, 0.2000, 0.7000],
              [0.2000, 0.9000, 0.4000, 0.5000, 0.7000]]]])
    target = torch.tensor([[[0., 2., 1., 1., 1.],
             [0., 1., 0., 1., 1.],
             [1., 2., 0., 2., 0.],
             [2., 1., 1., 0., 1.],
             [1., 2., 1., 2., 2.]]])
    PA = Metric.PA_old(output, target)
    print(PA)
    PA = Metric.PA(output,target, 0.6)
    print(PA)



if __name__ == "__main__":
    import fire

    fire.Fire(main)