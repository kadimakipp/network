#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: wind.py
@time: 2019/9/29 上午11:18
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from torchvision.models.resnet import resnet18 as ResNet18
import matplotlib.pyplot as plt
import pandas as pd
import math

"""
PyTorch lr scheduler  plot it
"""

class Wind(object):
    def __init__(self):
        self.net = ResNet18()
        self.optim = optim.SGD(self.net.parameters(), lr = 1,momentum=0.9)
        plt.figure()


    def StepLR(self, step_size,gamma=0.1):
        """
        等间隔调整学习率，调整倍数为gamma倍，调整间隔是step
        """
        return optim.lr_scheduler.StepLR(self.optim,
                                         step_size=step_size,
                                         gamma=gamma)

    def MultiStepLR(self, milestones, gamma=0.1):
        """按制定的间隔调整学习率，milestones为递增的list"""
        return optim.lr_scheduler.MultiStepLR(self.optim,
                                              milestones=milestones,
                                              gamma=gamma)

    def ExponentialLR(self, gamma):
        """按指数衰减调整学习率，调整公式:lr = lr*gamma**epoch"""
        return optim.lr_scheduler.ExponentialLR(self.optim,
                                                gamma=gamma)

    def CosineAnnealingLR(self,T_max: int, eta_min=1e-8):
        """
        以余弦函数为周期，幷在每个周期最大值时重新设置学习率
        paper:1608.03983.pdf
        T_max(int),一个学习率周期的迭代次数，即T_max个epoch后重新设置学习率
        eta_min(float)-最小学习率
        n_t = n_min + 1/2(n_max - n_min)(1+cos(T_cur/T_max*PI)
        """
        return optim.lr_scheduler.CosineAnnealingLR(self.optim,
                                                    T_max=T_max,
                                                    eta_min=eta_min)

    def ReduceLROnPlateau(self):
        """
        当某个指标不在变化(下降或升高)，调整学习率。
        mode(str)- 有min,max两个模式，min表示指标不在降低，max表示指标不在增大
        factor(float)-学习率倍数调整(等同于其它方法的gamma)lr = lr*factor
        patience(int)-(直译)耐心的意思，即该忍受多少个step不变化。注意可以是不连续的
        verbose(bool)- 是否打印学习率信息
        threshold-配合threshold mode使用
        threshold_mode(str)
        cooldown(int) 冷却时间，当调整学习率之后，让模型运行一段时间后，在开启监测模式
        min_lr(float or list)学习率下限
        eps(float) 学习率衰减的最小值，当学习率变化小于eps时，则不调整学习率。
        """
        #指标不容易模拟。不在example
        pass

    def CyclicLR(self):
        #paper 1506.01186
        pass

    def LambdaLR(self):
        """yourself define lr scheduler """
        def lambdalr(epoch):
            return epoch
        return optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lambdalr)

    def WarmRestartLR(self,lr, T_max:int, mult:int, eta_min:float, factor:float, restart=False):
        """ T_max: 周期， mutl 周期的比值, eta_min 最小lr, factor, 幅度缩小比例
        paper: https://arxiv.org/pdf/1608.03983.pdf
        """
        def cosineAnnealing(epoch):
            Tnum = epoch//T_max
            Tcur = epoch
            if restart:
                Tcur = epoch%T_max
            else:
                if Tnum!=0 and Tnum % 2==0:
                    Tnum  = Tnum-1
            T = T_max
            cos_in = Tcur / T * np.pi
            n_t = eta_min + factor**Tnum *(lr-eta_min)*(1+np.cos(cos_in))/2
            return n_t
        return optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=cosineAnnealing)

    def WarmRestartLR2(self, lr, T_max, mult,eta_min,factor):
        """new ways to warm restart"""
        def warmRestart(epoch):
            Tnum = epoch // T_max
            if mult>1:
                index = int(math.log(Tnum*(mult-1)+1,mult))
                Ts = [np.power(mult, i) for i in np.arange(0, 10)]
                T = Ts[index]*T_max
                f = np.power(factor, index)
                Tbe = 0
                while index >0:
                    index-= 1
                    Tbe += Ts[index]*T_max
                Tcur = (epoch - Tbe) / T* np.pi
                n_t = eta_min + f * (lr - eta_min) * (1 + np.cos(Tcur)) / 2
                return n_t
            else:
                f = np.power(factor,Tnum-1)
                cos_in = epoch%T_max*np.pi#key 减去前面的
                n_t = eta_min+f*(lr-eta_min)*(1+np.cos(cos_in))/2
                return n_t
        return optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=warmRestart)

    def Lmbda_warm_restart(self):
        from samhi.auxiliary import AuxFunction as AuxF
        return AuxF.WarmRestart(self.optim)

    def Lmbda_cosine_annealing(self):
        from samhi.auxiliary import AuxFunction as AuxF
        return AuxF.CosineAnnealing(self.optim)

    def show(self, epochs, scheduler=None):
        lrs = []
        for epoch in range(epochs):
            if scheduler != None:
                scheduler.step()
            lr = 0
            for p in self.optim.param_groups:
                lr = p['lr']
            lrs.append(lr)
            self.optim.zero_grad()
        plt.plot(range(epochs), lrs)
        plt.show()

def main():
    wind = Wind()
    # wind.show(300, wind.StepLR(50,0.5))
    # wind.show(120, wind.MultiStepLR(milestones=[20,60,100]))
    # wind.show(30, wind.ExponentialLR(0.1))  #学习率下降速度超快。
    # wind.show(150, wind.CosineAnnealingLR(30,0.01))
    # wind.show(150, wind.LambdaLR())#ConsineAnnealingLR, need change
    # wind.show(150, wind.WarmRestartLR(0.1, 30, 2, 0.001, 0.65))
    # wind.show(150, wind.WarmRestartLR2(0.1, 30,2,0.001,0.65))
    wind.show( 150, wind.Lmbda_cosine_annealing())
    wind.show(150, wind.Lmbda_warm_restart())
if __name__ == "__main__":
    import fire

    fire.Fire(main)