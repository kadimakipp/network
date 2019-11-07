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
from dataset.hed_bsds import hedBSDS
from net.Unet import UNet
import os
from tensorboardX import SummaryWriter

class Alchemy(object):
    def __init__(self,image_size=(3,224,224),classes=1,lr=0.01):
        self.gourd = os.path.join(AuxF.project_path(), 'gourd')
        self.boy = os.path.join(AuxF.project_path(), 'boy')
        self.num_class = classes
        self.image_size = image_size
        hedbsds = hedBSDS()
        self.loader = hedbsds.get_loader(1,self.image_size[1])
        print('init dataset done...')
        self.device = AuxF.device()
        self.net = UNet(self.image_size[0], classes)
        self.net.to(self.device)
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        self.lr_scheduler = self.LRScheduler()
        self.summary = SummaryWriter(logdir=AuxF.log_name(self.boy, 'unet'))

    def LRScheduler(self):
        return AuxF.WarmRestart(self.optimizer,
                                    T_max=60*len(self.loader), factor=0.75)

    def Train(self, epochs):
        data_length = len(self.loader)
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
                if i%2==0:
                    n_iter = epoch*data_length+i+1
                    self.summary.add_scalar('unet/loss', loss.item(),n_iter)
                    self.summary.add_scalar('unet/lr', AuxF.get_lr(self.optimizer), n_iter)
                    print('loss', loss.item())
                if i%4 == 0:
                    n_iter = epoch*data_length+1+i
                    im = torchvision.utils.make_grid(image.detach().cpu(), normalize=True)
                    self.summary.add_image('unet/image', im, n_iter)
                    hot = torchvision.utils.make_grid(out.detach().cpu(),normalize=True)
                    self.summary.add_image('unet/final', hot, n_iter)


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

def main(train='train'):
    torch.cuda.empty_cache()
    unet = Alchemy()
    if train in ['train']:
        unet.Train(180)
    else:
        unet.load_model()
        #unet.hot_map()


if __name__ == "__main__":
    import fire
    fire.Fire(main)