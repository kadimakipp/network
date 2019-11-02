#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: losses.py
@time: 2019/11/2 下午5:33
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torchvision
import torch.nn as nn
import math
import warnings
warnings.filterwarnings("ignore")
EXP = math.e

def L1_fun():
    input = torch.ones(3,5,dtype=torch.float32)*2
    target = torch.ones(3,5,dtype=torch.float32)
    #becase backward this is a scale
    loss = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
    output = loss(input, target)
    return output.numpy()

def MSE_fun():
    input = torch.ones(3, 5, dtype=torch.float32) * 3
    target = torch.ones(3, 5, dtype=torch.float32)
    # becase backward this is a scale
    loss = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    output = loss(input, target)
    return output.numpy()

def CrossEntropy_fun():
    loss_f = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=False, reduction='mean')
    input = torch.ones(3,5)*0.5 #batchsize=3,classes=5,neural=0.5
    target=torch.from_numpy(np.array([0,1,2])).type(torch.LongTensor)
    output= loss_f(input, target)



    #numpy
    batch = input[0].detach().numpy()
    target_1 = target[0].numpy()
    #第一项
    x_class = batch[target_1]
    #第二项
    sigma_exp_x = 0
    for o in batch:
        sigma_exp_x += pow(EXP, o)

    log_sigma_exp_x = math.log(sigma_exp_x)

    #两项相加
    loss_1 = -x_class+log_sigma_exp_x
    print('第一个样本的loss:', loss_1)


    #------------------weight
    weight = torch.from_numpy(np.array([0.1,0.2,0.3,0.4,0.5])).float()
    loss_weight = nn.CrossEntropyLoss(weight=weight, reduce=False)
    output2 = loss_weight(input, target)

    print('原始loss值为',output2)
    print('第一个样本是0类,weight=0.1,输出为1.6094*0.1=', 0.16094)
    #-------------------ignore_index
    loss_ignore_1 = nn.CrossEntropyLoss(reduce=False,ignore_index=1)
    loss_ignore_2 = nn.CrossEntropyLoss(reduce=False,ignore_index=2)
    loss_1 = loss_ignore_1(input, target)
    loss_2 = loss_ignore_2(input, target)
    print('ignore_index=1',loss_1)#注意时class为1，并不是index为1
    print('ignore_index=2',loss_2)#注意时class为1，并不是index为1
    return output

def NLL_fun():
    #-----negative log liklihood loss
    weight = torch.from_numpy(np.array([0.6, 0.2, 0.2])).float()
    input = torch.from_numpy(np.array([[0.7,0.2,0.1],[0.4,1.2,0.4]])).float()
    target = torch.from_numpy(np.array([0, 0])).type(torch.LongTensor)

    loss_f = nn.NLLLoss(weight=weight, size_average=True, reduce=False)
    output = loss_f(input, target)
    # pytorch example:
    m = nn.LogSoftmax(dim=1)
    loss_f = nn.NLLLoss(reduce=False)
    #input size NxC=3x5
    input = torch.randn(3,5)
    target = torch.tensor([1,0,4])
    print('-----------1D---------------------')
    print('input:', m(input))
    print('target:', target)
    print('1D:',loss_f(m(input), target))

    # 2D loss example (used, for example, with image input)
    N, C = 1,2
    loss_f = nn.NLLLoss(reduce=False)
    data = torch.randn(N, C, 3, 3)
    m = nn.LogSoftmax(dim=1)
    target = torch.empty(N, 3,3,dtype=torch.long).random_(0,C)
    input = m(data)
    print('----------------2D-----------------')
    print('input: ', input)
    print('target', target)
    print('2D result:', loss_f(input, target))
    return output

def PoissonNLL_fun():
    log_input = torch.randn(5,2)
    target = torch.randn(5,2)
    loss_f = nn.PoissonNLLLoss()
    return loss_f(log_input,target)

def KLdiv_fun():
    loss_f = nn.KLDivLoss(size_average=False,reduce=False)
    loss_f_mean = nn.KLDivLoss(size_average=True, reduce=True)
    input = torch.from_numpy(np.array([[0.1132,0.5477, 0.3390]])).float()
    target = torch.from_numpy(np.array([[0.8541, 0.0511, 0.0947]])).float()

    output = loss_f(input, target)
    loss_mean = loss_f_mean(input, target)
    print('mean: ', loss_mean)
    #-------------------收到计算样本的第一个元素的loss,注意这里只有一个样本,是element-wise计算的。
    raw = input[0].numpy()
    raw_1 = raw[0]
    target_1 = target[0][0].numpy()
    loss_1 = target_1*(np.log(target_1) - raw_1)
    print('第一个样本第一个元素的loss：', loss_1)

    return output



def main(l='NLL'):
    output = None
    if l in ['L1']:
        output = L1_fun()
    elif l in ['MSE']:
        output = MSE_fun()
    elif l in ['CrossEntropy']:
        output = CrossEntropy_fun()
    elif l in ['NLL']:
        output = NLL_fun()
    elif l in ['KLDiv']:
        output = KLdiv_fun()
    elif l in ['PoissonNLL']:
        output = PoissonNLL_fun()
    else:
        print('l must be in [L1, MSE, CrossEntropy, NLL, KLDiv, PoissonNLL]')
    print('\n\n')
    print(output)






if __name__ == "__main__":
    import fire

    fire.Fire(main)