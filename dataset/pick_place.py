#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: pick_place.py
@time: 2019/11/6 上午10:38
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as TransForms
from PIL import Image
from torch.utils.data import Dataset
import os
import cv2

'''
pick and place dataset.md only suction way
'''
DEP_MEAN = 0.4313711812112666
DEP_STD = 0.06876936556987262
class Suction(Dataset):
    def __init__(self, root, transform, train='train', use_im=False):
        super(Suction, self).__init__()
        self.root = root
        self.transform = transform
        self.use_im = use_im
        index = os.path.join(self.root, 'train-split.txt')
        if train not in ['train']:
            index = os.path.join(self.root, 'test-split.txt')

        with open(index,'r') as f:
            self.ids = f.readlines()
            self.ids = [s.strip('\n') for s in self.ids]
            f.close()

        self.__color = os.path.join(self.root, 'color-input')
        self.__depth = os.path.join(self.root, 'depth-input')
        self.__depth_back = os.path.join(self.root, 'depth-background')
        self.__label = os.path.join(self.root, 'label')
        self.__intrinsics = os.path.join(self.root, 'camera-intrinsics')

    def __get_intrinsics(self, root):
        with open(root, 'r') as f:
            intr = f.readlines()
            intr = [s.strip('\n').split('\t') for s in intr]
            intr = [l[:3] for l in intr[:3]]
            intr = [float(s)  for l in intr for s in l]
            intrinsics = np.array(intr).reshape(3,3)
            f.close()
        return intrinsics

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        label_path = os.path.join(self.__label, id+'.png')
        label = Image.open(label_path)
        label = np.array(label)
        label[label==128] = 1
        label[label==255] = 2
        label = torch.from_numpy(label)

        depth_path = os.path.join(self.__depth, id+'.png')
        depth_back_path = os.path.join(self.__depth_back, id+'.png')
        depth = Image.open(depth_path)
        depth = np.array(depth).astype(np.float)/1e+4
        mask = (depth>0)&(depth<0.9)
        mask = mask.astype(np.float)
        mask[mask==0.0] = -1.0
        mask = torch.from_numpy(mask).unsqueeze(0)
        #depth_back = Image.open(depth_back_path)
        #depth_back = np.array(depth_back).astype(np.float)/1e+4

        depth = (depth-DEP_MEAN)/DEP_STD
        # depth = depth - depth_back

        depth = torch.from_numpy(depth).unsqueeze(0)
        cat = torch.cat((depth, mask))
        sample = {'depth': depth, 'label': label}
        sample['mask'] = mask
        sample['cat'] = cat


        if self.use_im:
            im_path = os.path.join(self.__color, id+'.png')
            im = Image.open(im_path)
            im = self.transform(im)
            sample['image'] = im
        return sample

class Pick(object):
    def __init__(self):
        self.root = '/media/kipp/work/Datas/Pick-and-Place/suction_data'
        self.shuffle = True
        self.num_work = 4

    def Transform(self, img_size):
        transform = TransForms.Compose([
            TransForms.ToTensor(),
            TransForms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform


    def get_loader(self, batch_size, img_size, mode='train', use_im=True):
        if mode not in ['train']:
            self.shuffle=False
        transform= self.Transform(img_size)
        return torch.utils.data.DataLoader(
            Suction(self.root, transform=transform,train=mode, use_im=use_im),
            batch_size=batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_work
        )

import matplotlib.pyplot as plt

def check_loader():
    plt.figure()
    pick = Pick()
    loader = pick.get_loader(1, 448, mode='train')
    print(len(loader))
    for i, samples in enumerate(loader):
        images, depths, labels = samples['image'], samples['depth'], samples['label']
        print(images.shape, depths.shape, labels.shape)
        dis_img = images[0].numpy().transpose(1, 2, 0)
        dis_img = dis_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        dis_img = dis_img * 255
        plt.imshow(dis_img.astype(np.uint8))
        plt.show()
        dis_lab = labels[0].numpy() * 127
        plt.imshow(dis_lab.astype(np.uint8))
        plt.show()
        dis_dep = depths[0].numpy().squeeze()#DEP_STD+DEP_MEAN
        plt.imshow(dis_dep)
        plt.show()
        mask = samples['mask']
        dis_mask = mask[0].numpy().squeeze()
        plt.imshow(dis_mask)
        plt.show()
        cat = samples['cat']
        print(cat.shape)
        break

def compute_depth_mean_std(root):
    index = os.path.join(root, 'train-split.txt')
    __depth = os.path.join(root, 'depth-input')
    with open(index, 'r') as f:
        ids = f.readlines()
        ids = [os.path.join(__depth,s.strip('\n')+'.png') for s in ids]
        f.close()
    im_mean = []
    for d in ids:
        depth = Image.open(d)
        print(d)
        depth = np.array(depth).astype(np.float)/1e+4
        im_mean.append(depth.mean())

    else:
        im_mean =np.array(im_mean)
        mean = im_mean.mean()
        std = np.power(np.power(im_mean-mean, 2).mean(), 0.5)
        print("DEP_MEAN = {} \nDEP_STD = {}".format(mean, std))


def main(code = "train"):
    if code in ['compute']:
        root = '/media/kipp/work/Datas/Pick-and-Place/suction_data'
        compute_depth_mean_std(root)
    else:
        check_loader()




if __name__ == "__main__":
    import fire
    fire.Fire(main)