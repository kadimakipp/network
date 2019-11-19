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

    @classmethod
    def device(cls):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @classmethod
    def to_categorical(cls,y, num_columns):
        """Return one-hot encoded device variable"""
        y_cat = torch.zeros((y.shape[0]), num_columns)
        y_cat[range(y.shape[0]), y] = 1.0

        return y_cat.to(device=AuxFunction.device())

    @classmethod
    def update_lr(cls,optimizer, lr):
        for params in optimizer.param_groups:
            params['lr'] = lr

    @classmethod
    def get_lr(cls,optimizer):
        lr = 0
        for params in optimizer.param_groups:
            lr = params['lr']
        return lr

    @classmethod
    def parameters_total(cls,model):
        total = sum(param.numel() for param in model.parameters())
        return total / 1e6

    @classmethod
    def WarmRestart(cls,optim, lr=1, T_max=10, mult=2,eta_min=0.00001,factor=0.65):
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

    @classmethod
    def CosineAnnealing(cls, optim, lr=1, T_max=30,eta_min=0.0001,factor=0.65, restart=False):
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

    @classmethod
    def dir_name(cls, dir_name,net_name,comment=''):
        name = os.path.join(dir_name, net_name)
        import socket
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        logdir = os.path.join(
            name, current_time + '_' + socket.gethostname() + comment)
        return logdir

    @classmethod
    def project_path(cls):
        return os.path.dirname(os.path.dirname(__file__))

    @classmethod
    def make_dirs(cls, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    @classmethod
    def dir_gourd2boy(cls, dir):
        return dir.replace('gourd','boy')
    @classmethod
    def dir_boy2gourd(cls,dir):
        return dir.replace('boy','gourd')



if __name__ == "__main__":
    aux = AuxFunction()
    pro_dir = aux.project_path()
    boy_dir = aux.dir_name(os.path.join(pro_dir, 'boy'),'my_net')
    gourd_dir = aux.dir_boy2gourd(boy_dir)
    print(boy_dir, gourd_dir)
    aux.make_dirs(gourd_dir)
    aux.make_dirs(boy_dir)



