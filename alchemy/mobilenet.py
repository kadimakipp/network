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
from net.mobilenet import MobileNetV2
import os

class Alchemy(object):
    def __init__(self, num_class, lr=0.005):
        self.num_class = num_class
        self.device = AuxF.device()
        self.net = MobileNetV2(num_class)
        self.net.to(self.device)
        self.lr = lr
        print("init net")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.imagenet = miniImagenet()
        self.loader = self.imagenet.get_loader(16,224,"train")
        print("init data")

    def Train(self, epochs):
        cur_lr = self.lr
        for epoch in range(epochs):
            for i, (images, labels) in enumerate(self.loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                out= self.net(images)
                loss = self.criterion(out, labels)
                loss.backward()
                self.optimizer.step()
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} Lr: {:e}"
                      .format(epoch + 1, epochs, i + 1, len(self.loader), loss.item(), cur_lr))
                if i+1 == len(self.loader):
                    self.save_model(loss, epoch)
                    out_v = out.detach().data
                    _, predicted = torch.max(out_v, 1)
                    total = labels.size(0)
                    correct = (predicted == labels).sum().item()
                    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
            if (epoch+1)%10 ==0:
                cur_lr /=10
                AuxF.update_lr(self.optimizer, cur_lr)


    def save_model(self, loss, epoch):
        checkpoint = {
            'loss': loss,
            'epoch': epoch,
            'model': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        model_path = os.path.join('../gourd', 'checkpoint-%03d.pth.tar' % (epoch))
        torch.save(checkpoint, model_path)
        # copy
        # import shutil
        # shutil.copy('checkpoint.pth.tar', model_path)

    def load_model(self):
        model_path = os.path.join('../gourd', 'checkpoint-029.pth.tar')
        assert os.path.isfile(model_path)
        checkpoint = torch.load(model_path)
        best_acc = checkpoint['loss']
        start_epoch = checkpoint['epoch']
        self.net.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print('Load checkpoint at epoch %d.' % start_epoch)



    def val_model(self):
        test_loader = self.imagenet.get_loader(16,224,"val&test")
        self.net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))



# tonight complete the project
# Refer[https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py]

def main():
    torch.cuda.empty_cache()
    mobilenet = Alchemy(10)
    mobilenet.Train(30)
    #mobilenet.load_model()
    mobilenet.val_model()
if __name__ == "__main__":
    import fire

    fire.Fire(main)