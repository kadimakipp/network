#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: auxiliary.py
@time: 2019/9/18 下午8:00
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torchvision
import os

class AuxFunction(object):
    def __init__(self):
        pass

    @staticmethod
    def device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def to_categorical(y, num_columns):
        """Return one-hot encoded device variable"""
        y_cat = torch.zeros((y.shape[0]), num_columns)
        y_cat[range(y.shape[0]), y] = 1.0

        return y_cat.to(device=AuxFunction.device())

    @staticmethod
    def update_lr(optimizer, lr):
        for params in optimizer.param_groups:
            params['lr'] = lr

    @staticmethod
    def parameters_total(model):
        total = sum(param.numel() for param in model.parameters())
        return total / 1e6

import matplotlib.pyplot as plt
import pandas as pd

class FireInfo(object):
    def __init__(self):
        root = os.path.dirname(__file__)
        self.fire = os.path.join(root, "fire")
        plt.figure()
        self.loss= list([0])
        self.acc = list([0])
        self.lr = list([0])

    def update(self, loss, acc,lr):
        self.loss.append(loss)
        self.acc.append(acc)
        self.lr.append(lr)

    def clear(self):
        self.loss = list([0])
        self.acc = list([0])
        self.lr = list([0])
    def save(self):
        data = {'loss': self.loss, 'acc': self.acc, 'lr':self.lr}
        csv = pd.DataFrame(data)
        csv.to_csv(os.path.join(self.fire, 'fire.csv'))

    def read(self):
        self.clear()
        data = pd.read_csv(os.path.join(self.fire, 'fire.csv'))
        loss = data['loss']
        self.loss.extend(loss.to_list())
        acc = data['acc']
        self.acc.extend(acc.to_list())
        lr = data['lr']
        self.lr.extend(lr.to_list())

    def display(self):
        ax1 = plt.subplot(3,1,1)
        ax1.set_title("Loss")
        plt.plot(np.arange(len(self.loss)), self.loss,color='r')
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.set_title("Acc")
        plt.plot(np.arange(len(self.acc)), self.acc,color='r')
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.set_title("LR")
        plt.plot(np.arange(len(self.lr)), self.lr,color='r')
        #plt.draw()
        plt.show()


if __name__ == "__main__":
    fire = FireInfo()
    for i in np.arange(1,100):
        fire.update(i,i,i)
    fire.save()
    fire.read()
    fire.display()

