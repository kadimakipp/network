#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: coco.py
@time: 19-11-5 下午11:24
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
from pycocotools.coco import COCO as COCOtool


class COCO(Dataset):
    def __init__(self, root, transform, target_transform,train='train'):
        super(COCO, self).__init__()
        self.root = root
        self.annFile = os.path.join(self.root, 'annotations/instances_minival2014.json')
        self.coco = COCOtool(self.annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        """
        :param index:
        :return:Tuple(image, target)
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        return img, target

    def __len__(self):
        return len(self.ids)

class CoCo(object):
    def __init__(self):
        self.root = '/media/kipp/data/DATASET/COCO'
        self.num_work = 4
        self.shuffle = True

    def Transform(self, img_size):
        transform = [
            # transforms.RandomCrop(224),
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomAffine(5),
            transforms.Resize((img_size, img_size), Image.BICUBIC),
            transforms.ToTensor()
        ]
        return transform


    def get_loader(self, batch_size, img_size, mode='train'):
        transform= self.Transform(img_size)
        return torch.utils.data.DataLoader(
            COCO(self.root, transform=transform,target_transform=transform,train=mode),
            batch_size=batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_work
        )

def main():
    coco = CoCo()
    loader = coco.get_loader(1,224, 'train')
    print(len(loader))

if __name__ == "__main__":
    import fire

    fire.Fire(main)