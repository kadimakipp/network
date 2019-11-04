#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: Unet.py
@time: 2019/11/4 上午11:05
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

class ConvEncode(nn.Sequential):
    def __init__(self,inc,ouc,kernel=3):
        super(ConvEncode, self).__init__(
            nn.Conv2d(kernel_size=kernel, in_channels=inc,out_channels=ouc,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(ouc),
            nn.Conv2d(kernel_size=kernel,in_channels=ouc,out_channels=ouc,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(ouc),
        )

class ConvDecode(nn.Sequential):
    def __init__(self,inc,mic,ouc,kernel=3):
        super(ConvDecode, self).__init__(
            nn.Conv2d(kernel_size=kernel, in_channels=inc,out_channels=mic, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mic),
            nn.Conv2d(kernel_size=kernel,in_channels=mic,out_channels=mic,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mic),
            nn.ConvTranspose2d(in_channels=mic,out_channels=ouc,
                               kernel_size=3,stride=2,padding=1,output_padding=1)
        )

class FinalBlock(nn.Sequential):
    def __init__(self,inc, mic,ouc,kernel=3):
        super(FinalBlock, self).__init__(
            nn.Conv2d(in_channels=inc, out_channels=mic, kernel_size=kernel,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mic),
            nn.Conv2d(in_channels=mic, out_channels=mic, kernel_size=kernel,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mic),
            nn.Conv2d(in_channels=mic, out_channels=ouc, kernel_size=kernel,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(ouc)
        )

class UNet(nn.Module):
    def __init__(self, inc, classes):
        super(UNet, self).__init__()
        self.down_1 = ConvEncode(inc=inc,ouc=64)
        self.down_2 = ConvEncode(inc=64,ouc=128)
        self.down_3 = ConvEncode(inc=128, ouc=256)
        self.down_sample = nn.MaxPool2d(kernel_size=2)
        self.bottleneck = ConvDecode(inc=256,mic=512,ouc=256)
        self.up_3 = ConvDecode(512,256,128)
        self.up_2 = ConvDecode(256,128,64)
        self.final = FinalBlock(128,64,classes)

    def crop_and_concat(self, up, down, crop=False):
        """
        This layer crop the layer from contraction block and concat it with
        expansive block vector
        """
        if crop:
            c = (down.size()[2]-up.size()[2])//2
            down = F.pad(down, (-c,-c,-c,-c))
        return torch.cat((up,down),1)

    def forward(self, x):
        level_1 = self.down_1(x)
        downsample_1 = self.down_sample(level_1)
        level_2 = self.down_2(downsample_1)
        downsample_2 = self.down_sample(level_2)
        level_3 = self.down_3(downsample_2)
        downsample_3 = self.down_sample(level_3)

        bottleneck1 = self.bottleneck(downsample_3)

        concat_level_3 = self.crop_and_concat(bottleneck1,level_3,crop=True)
        up_2 = self.up_3(concat_level_3)
        concat_level_2 = self.crop_and_concat(up_2, level_2, crop=True)
        up_1 = self.up_2(concat_level_2)
        concat_level_1 = self.crop_and_concat(up_1, level_1, crop=True)
        output = self.final(concat_level_1)
        return output

def main():
    unet = UNet(3, 2)
    input = torch.randn([1,3,224,224])
    out = unet(input)
    print(unet)




if __name__ == "__main__":
    import fire

    fire.Fire(main)