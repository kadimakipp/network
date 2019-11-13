#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: metrics.py
@time: 2019/11/13 上午10:53
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
multi class Metrics Benchmark
1. generator confusion matrix.
2. statistical information

-----------------------------------------------
confusionMatrix
P\L     Ture_1    Ture_2   Ture_3
Pred_1     5        2        1
Pred_2     3        10       6
Pred_3     0        2        5

k元分类的混淆矩阵就是k乘k的矩阵。矩阵(i, j)上的数字就是模型预测为分类i、但实际为分类j的样本的个数。
P_ij  i为GT j为pred 表示属于i类预测为j类。
P_ii  TP(True positive) 
P_ij  FP(False positive)
P_ji  FN(False negative)
P_jj  TN(True negative)
注意对角线、行和列表示的物理意义
"""
class MultiClassesMetrics(object):
    def __init__(self, n_classes):
        self.__n_classes = n_classes
        self.confusion_matrix = np.zeros((self.__n_classes,)*2)

    def add_batch(self, pred, gt):
        assert pred.shape==gt.shape
        self.confusion_matrix += self._generate_matrix(pred, gt)

    def reset(self):
        self.confusion_matrix = np.zeros((self.__n_classes,)*2)

    def _generate_matrix(self, pred, gt):
        mask = (gt>=0)&(gt<self.__n_classes)
        label = self.__n_classes*gt[mask].astype('int')+pred[mask]
        count = np.bincount(label,minlength=self.__n_classes**2)
        confusion_matrix = count.reshape(self.__n_classes, self.__n_classes)
        return confusion_matrix

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum()/self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix)/self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix)/(
            np.sum(self.confusion_matrix, axis=1)+np.sum(self.confusion_matrix, axis=0)-
            np.diag(self.confusion_matrix)
        )
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1)/np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix)/(
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix)
        )
        FWIoU = (freq[freq>0]*iu[freq>0]).sum()
        return FWIoU


def main():
    batch_size = 16
    classes = 3
    metric = MultiClassesMetrics(classes)
    #classification
    pred = np.random.randint(classes, size = (batch_size,1))
    tar = np.random.randint(classes, size = (batch_size, 1))
    print(np.concatenate((pred, tar), axis=1))
    metric.add_batch(pred, tar)
    PA = metric.Pixel_Accuracy()
    MPA = metric.Pixel_Accuracy_Class()
    MIoU = metric.Mean_Intersection_over_Union()
    FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
    print('PA: {}\nMPA: {}\nMIoU: {}\nFWIoU: {}\n'.format(PA,MPA,MIoU,FWIoU))

    pred = np.random.randint(classes, size=(batch_size,5,5))
    tar = np.random.randint(classes, size=(batch_size, 5,5))
    metric.reset()
    metric.add_batch(pred, tar)
    PA = metric.Pixel_Accuracy()
    MPA = metric.Pixel_Accuracy_Class()
    MIoU = metric.Mean_Intersection_over_Union()
    FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
    print('PA: {}\nMPA: {}\nMIoU: {}\nFWIoU: {}\n'.format(PA,MPA,MIoU,FWIoU))




if __name__ == "__main__":
    import fire
    fire.Fire(main)