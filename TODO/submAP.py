#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: submAP.py
@time: 19-11-12 下午9:48
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

"""
TODO: merge to Metric
"""
"""
multi class mAP
VOC mAP Castration Edition

"""
import matplotlib.pyplot as plt
def multi_class_mAP(output, target):
    """
    [reference]:
    https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173
    https://medium.com/@hfdtsinghua/calculate-mean-average-precision-map-for-multi-label-classification-b082679d31be
    https://github.com/luliyucoordinate/eval_voc/blob/master/eval_voc.py
    """
    #TODO:
    pass
# No threshold
def PR(target, positive):
    """
    P = TP/(TP+FP)
    R = TP/(TP+FN)=TP/positive
    """
    TP = target==1
    TP = TP.sum().float()
    TP_add_FP = target.shape[0]
    P = TP/TP_add_FP
    R = TP/positive
    return P,R

def AP(target):
    """
    Average Precision
    """
    positive = target==1
    positive = positive.sum().float()
    precision=[]
    recall=[]
    for i in range(1,target.shape[0]+1):
        label = target[:i]
        P,R = PR(label, positive)
        precision.append(P.numpy())
        recall.append(R.numpy())
    # # smooth
    for i in range(len(precision)-1)[::-1]:
        precision[i] = max(precision[i+1], precision[i])

    #calc AP, PR cure must be on the point(0,0)
    #method 1: error too big
    ap = 0
    for i in range(len(precision)-1)[::-1]:
        delta_x = recall[i+1]-recall[i]
        ap += delta_x*precision[i+1]
    else:
        # this moment i = 0
        delta_x = recall[i] - 0
        ap += delta_x*precision[i]
    #method 2: another function for
    return ap, precision, recall


def mAP(output, target, plot=False):
    """
    mAP mean average precision
    """
    # sort
    output, indices = torch.sort(output, dim=0, descending=True)
    #get each class AP
    classes = target.shape[1]
    aps = []
    if plot:
        plt.figure()
    for i in range(classes):
        label = target[:,i][indices[:,i]]
        ap, p, r = AP(label)
        aps.append(ap)
        if plot:
            plt.plot(np.array(r), np.array(p), '-o')
    aps = np.array(aps)
    map = aps.mean()
    if plot:
        plt.show()
    return map, aps


def main():
    output = torch.tensor([[0.2,0.6,0.1,0.8],
                           [0.4,0.9,0.8,0.6],
                           [0.8,0.4,0.5,0.7]])
    target = torch.tensor([[0,1,1,1],
                           [0,0,1,0],
                           [1,1,0,0]])
    map, aps = mAP(output,target, plot=True)
    print(map,'\n',aps)
    #example 2
    output = torch.tensor([[0.3408, 0.1951, 0.2455, 0.2186],
                           [0.2531, 0.2994, 0.2367, 0.2108],
                           [0.2008, 0.3744, 0.2102, 0.2146]])
    target = torch.tensor([[1, 0, 0, 1],
                           [1, 1, 0, 0],
                           [0, 1, 1, 1]])
    map, aps = mAP(output, target, plot=True)
    print(map,'\n',aps)

if __name__ == "__main__":
    import fire

    fire.Fire(main)