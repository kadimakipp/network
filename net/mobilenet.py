#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: mobilenet.py
@time: 2019/9/23 下午5:23
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
import torch.nn.functional as F
from net.SENet import SE

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3,stride=1,groups=1):
        padding = (kernel_size-1)//2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes,kernel_size,stride, padding,groups=groups,bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp,oup,stride,expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            #pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            #dw 3x3x1xhidden_dim kernerl
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            #pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.block = nn.Sequential(*layers)
        self.se = SE(inp)

    def forward(self, x):
        if self.use_res_connect:
            out = self.block(x)
            w = self.se(x)
            return x +out*w
            # out = self.block(x)
            # w = F.avg_pool2d(out, out.shape[2])
            # w = F.relu6(self.se_1(w), inplace=True)
            # w = F.sigmoid(self.se_2(w))
            # return x+out*w

        else:
            return self.block(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10,width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 16
        last_channel = 512
        inverted_residual_setting = [
            #t, c, n, s
            [1, 16, 1, 1],
            [2, 24, 1, 2],
            [2, 32, 1, 2],
            [2, 64, 2, 2],
            [2, 96, 1, 2],
            [2, 160, 1, 1],
            # #----------------
            # [1, 16, 1, 1],
            # [6, 24, 2, 2],
            # [6, 32, 3, 2],
            # [6, 64, 4, 2],
            # [6, 96, 3, 1],
            # [6, 160, 3, 2],
            # [6, 320, 1, 1],
        ]


        #head
        input_channel = int(input_channel*width_mult)
        self.last_channel = int(last_channel*max(1.0, width_mult))
        features = [ConvBNReLU(3,input_channel,stride=2)]
        #building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c*width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel,output_channel,stride,expand_ratio=t))
                input_channel = output_channel
        #build last several layers
        features.append(ConvBNReLU(input_channel,self.last_channel, kernel_size=1))

        self.features = nn.Sequential(*features)

        #building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        #weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


def main():
    # torchvision.models.MobileNetV2(10)
    from samhi.model_tools import ModelTools
    from samhi.torchsummary import summary
    net = MobileNetV2(10)
    summary(net, input_size=(3, 112, 112),device="cpu")
    # print(net)
    x = torch.randn(1,3,112,112,requires_grad=True)

    # tools = ModelTools(x, net)
    # tools.print_keras_summary_like()
    # tools.print_parameters_total()
    # # tools.print_model_flops()
    # print(y.shape)
    # print(y.grad_fn)
    # def for_hook(module, input, output):
    #     ms = module.named_modules()
    #     for id, m in enumerate(ms):
    #         print(id, '----' , m)
    #
    #
    #     #for inp in input:
    #     #    print(inp)
    #     #for oup in output:
    #     #    print(oup)
    # hook = net.register_forward_hook(for_hook)
    # y = net(x)
    # hook.remove()



if __name__ == "__main__":
    import fire
    fire.Fire(main)