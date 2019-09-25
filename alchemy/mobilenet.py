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
import os

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
        self.loader = imagenet.get_loader(64,224,"train")
        #self.val_loader = imagenet.get_loader(32,224,"val")
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
                    self.save_model(loss, epoch)
                    out_v = out.detach().data
                    _, predicted = torch.max(out_v, 1)
                    total = labels.size(0)
                    correct = (predicted == labels).sum().item()
                    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

    def save_model(self, loss, epoch):
        checkpoint = {
            'loss': loss,
            'epoch': epoch,
            'model': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        model_path = os.path.join('../weights', 'checkpoint-%03d.pth.tar' % (epoch))
        torch.save(checkpoint, model_path)
        # copy
        # import shutil
        # shutil.copy('checkpoint.pth.tar', model_path)

    def load_model(self):
        model_path = os.path.join('../weights', 'checkpoint-000.pth.tar')
        assert os.path.isfile(model_path)
        checkpoint = torch.load(model_path)
        best_acc = checkpoint['loss']
        start_epoch = checkpoint['epoch']
        self.net.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print('Load checkpoint at epoch %d.' % start_epoch)


# tonight complete the project
# Refer[https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py]

def main():
    mobilenet = MobileNetV2(10)
    mobilenet.Train(80)
    #mobilenet.load_model()
if __name__ == "__main__":
    import fire

    fire.Fire(main)