#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: hed_bsds.py
@time: 2019/11/1 下午4:43
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
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

"""
HED-BSDS dataset
Contour Detection and Image Segmentation Resources
link https://cseweb.ucsd.edu/~weijian/static/datasets/hed/hed-data.tar
"""

class HEDBSDS(Dataset):
    def __init__(self, root,transform,train='train'):
        super(HEDBSDS, self).__init__()
        self.train=train
        self.transform = transforms.Compose(transform)


        self.root_dir = root
        self.image_lst = []
        self.label_lst = []
        if train in ['train']:
            lst_path = os.path.join(self.root_dir,'train_pair.lst')
            with open(lst_path, 'r') as f:
                self.data = f.readlines()
                for s in self.data:
                    s = s.replace('\n','')
                    i,l = s.split(' ')
                    self.image_lst.append(i)
                    self.label_lst.append(l)
                f.close()

        elif train in ['test']:
            lst_path = os.path.join(self.root_dir, 'test.lst')
            with open(lst_path, 'r') as f:
                self.data = f.readlines()
                for s in self.data:
                    s = s.replace('\n','')
                    self.image_lst.append(s)
                f.close()
        else:
            print('train is not in [train, val, test]')

        #label augment
        self.ToTensor =transforms.Compose([transforms.ToTensor()])



    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image = self.image_lst[item]
        if self.train in ['train']:
            label = self.label_lst[item]
        elif self.train in ['test']:
            label = None
        else:
            label = None
            print("Error check image")
        image = Image.open(os.path.join(self.root_dir, image))
        image = self.transform(image)
        _,h,w = image.shape
        if label is not None:
            label = Image.open(os.path.join(self.root_dir, label)).convert('L')
            label = label.resize((h, w), Image.BILINEAR)
            label = np.array(label).astype(np.float32)
            label[label < 1] = 0.0
            label[label>=1] = 1.0

            label = self.ToTensor(label)
        return image, label

class hedBSDS(object):
    def __init__(self):
        self.root = '/media/kipp/Files/hed/data/HED-BSDS'
        self.num_work=4
        self.shuffle=True

    def Transform(self, img_size):
        transform = [
            # transforms.RandomCrop(224),
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomAffine(5),
            transforms.Resize((img_size, img_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        return transform


    def get_loader(self, batch_size, img_size, mode='train'):
        transform= self.Transform(img_size)
        return torch.utils.data.DataLoader(
            HEDBSDS(self.root, transform=transform,train=mode),
            batch_size=batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_work
        )

import matplotlib.pyplot as plt

def main():
    plt.figure()
    train = 'train'
    hedbsds =hedBSDS()
    loader = hedbsds.get_loader(1, 448, train)
    print(len(loader))
    for i, (images, labels) in enumerate(loader):
        if i > 0:
            break
        dis_img = images[0].numpy().transpose(1, 2, 0)
        dis_img = dis_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        dis_img = dis_img * 255
        plt.imshow(dis_img.astype(np.uint8))
        plt.show()
        lab_img = labels[0].numpy()
        print(lab_img.shape)
        plt.imshow(lab_img.reshape(448,448))
        plt.show()

if __name__ == "__main__":
    import fire

    fire.Fire(main)