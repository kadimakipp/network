#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: unet.py
@time: 2019/11/4 下午4:45
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
from samhi.auxiliary import AuxFunction as AuxF
from samhi.auxiliary import FireInfo
from dataset.hed_bsds import hedBSDS
from net.Unet import UNet
import os

class Alchemy(object):
    def __init__(self,image_size=(3,224,224),classes=1,lr=0.01):
        self.gourd = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'gourd')
        self.num_class = classes
        self.image_size = image_size
        hedbsds = hedBSDS()
        self.loader = hedbsds.get_loader(4,self.image_size[1])
        print('init dataset done...')
        self.device = AuxF.device()
        self.net = UNet(self.image_size[0], classes)
        self.net.to(self.device)
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        self.lr_scheduler = self.LRScheduler()

    def LRScheduler(self):
        return AuxF.WarmRestart(self.optimizer,
                                    T_max=60*len(self.loader), factor=0.75)

    def Train(self, epochs):

        for epoch in range(epochs):
            for i , (image, label) in enumerate(self.loader):
                image = image.to(self.device)
                label = label.to(self.device)
                self.optimizer.zero_grad()
                out = self.net(image)
                loss = self.criterion(out, label)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                if i%200==0:
                    print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                          .format(epoch+1, epochs, i+1, len(self.loader), loss.item()))
            if epoch%10 ==0:
                self.save_model(epoch)

    def save_model(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        model_path = os.path.join(self.gourd, 'unet/checkpoint-%03d.pth.tar' % (epoch))
        torch.save(checkpoint, model_path)

    def load_model(self):
        model_path = os.path.join(self.gourd, 'unet/checkpoint-030.pth.tar')
        assert os.path.isfile(model_path)
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        self.net.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print('Load checkpoint at epoch %d.' % start_epoch)

def main():
    torch.cuda.empty_cache()
    unet = Alchemy()
    unet.Train(180)


if __name__ == "__main__":
    import fire
    fire.Fire(main)