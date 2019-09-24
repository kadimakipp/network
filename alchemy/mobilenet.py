#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: mobilenet.py
@time: 2019/9/24 上午11:48
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
from dataset.mini_imagenet import miniImagenet
from samhi.auxiliary import AuxFunction as AuxF
from net.mobilenet import Net

class MobileNetV2(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.device = AuxF.device()
        self.net = Net(num_class)
        self.net.to(self.device)
        print("init net")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.005)

        imagenet = miniImagenet()
        self.loader = imagenet.get_loader(16,224,"train")
        self.val_loader = imagenet.get_loader(32,224,"val")
        print("init data")

    def Train(self, epochs):
        for epoch in range(epochs):
            for i, (images, labels) in enumerate(self.loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                out= self.net(images)
                loss = self.criterion(out, labels)
                loss.backward()
                self.optimizer.step()
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, epochs, i + 1, len(self.loader), loss.item()))
                if i+1 == len(self.loader):
                    for image_v, labels_v in self.val_loader:
                        correct = 0
                        total = 0
                        image_v = image_v.to(self.device)
                        labels_v = labels_v.to(self.device)
                        out_v = self.net(image_v)
                        _, predicted = torch.max(out_v.data, 1)
                        total += labels_v.size(0)
                        correct += (predicted == labels_v).sum().item()
                        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# tonight complete the project
# Refer[https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py]

def main():
    mobilenet = MobileNetV2(10)
    mobilenet.Train(80)

if __name__ == "__main__":
    import fire

    fire.Fire(main)