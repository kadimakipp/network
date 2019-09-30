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
import math

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
    def get_lr(optimizer):
        lr = 0
        for params in optimizer.param_groups:
            lr = params['lr']
        return lr

    @staticmethod
    def parameters_total(model):
        total = sum(param.numel() for param in model.parameters())
        return total / 1e6

    @staticmethod
    def WarmRestart(optim, lr=1, T_max=10, mult=2,eta_min=0.00001,factor=0.65):
        """ T_max: 周期， mutl 周期的比值, eta_min 最小lr, factor, 幅度缩小比例
            paper: https://arxiv.org/pdf/1608.03983.pdf
        """
        def warmRestart(epoch):
            Tnum = epoch // T_max
            if mult > 1:
                index = int(math.log(Tnum * (mult - 1) + 1, mult))
                Ts = [np.power(mult, i) for i in np.arange(0, 10)]
                T = Ts[index] * T_max
                f = np.power(factor, index)
                Tbe = 0
                while index > 0:
                    index -= 1
                    Tbe += Ts[index] * T_max
                Tcur = (epoch - Tbe) / T * np.pi
                n_t = eta_min + f * (lr - eta_min) * (1 + np.cos(Tcur)) / 2
                return n_t
            else:
                f = np.power(factor, Tnum - 1)
                cos_in = epoch % T_max * np.pi  # key 减去前面的
                n_t = eta_min + f * (lr - eta_min) * (1 + np.cos(cos_in)) / 2
                return n_t
        return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmRestart)

    @staticmethod
    def CosineAnnealing(optim, lr=1, T_max=30,eta_min=0.0001,factor=0.65, restart=False):
        """ factor version
        T_max: 周期， mutl 周期的比值, eta_min 最小lr, factor, 幅度缩小比例
        """
        def cosineAnnealing(epoch):
            Tnum = epoch // T_max
            Tcur = epoch
            if restart:
                Tcur = epoch % T_max
            else:
                if Tnum != 0 and Tnum % 2 == 0:
                    Tnum = Tnum - 1
            T = T_max
            cos_in = Tcur / T * np.pi
            n_t = eta_min + factor ** Tnum * (lr - eta_min) * (1 + np.cos(cos_in)) / 2
            return n_t

        return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=cosineAnnealing)




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

