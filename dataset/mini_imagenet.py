#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: mini_imagenet.py
@time: 2019/9/23 下午1:51
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
"""
there are 100 classes with 600 samples of 84×84 color images per class. These 100 classes are divided into 64, 16,
and 20 classes respectively for sampling tasks for meta-training, meta-validation, and meta-test.
"""

class MiniImagenet(Dataset):
    def __init__(self, root,transform,new_csv=True,train="train", ten_class = False):
        super(MiniImagenet, self).__init__()
        self.transform = transforms.Compose(transform)
        self.__img_dir = os.path.join(root, "images")
        if new_csv:
            root = os.path.join(root, "new-csv")
        __test_csv = os.path.join(root, "test.csv")
        __val_csv = os.path.join(root, "val.csv")
        __train_csv = os.path.join(root, "train.csv")

        if train=="train":
            self.csv = pd.read_csv(__train_csv)
        elif train=="val":
            self.csv = pd.read_csv(__val_csv)
        elif train=="test":
            self.csv = pd.read_csv(__test_csv)
        elif train=="val&test":
            csv_val = pd.read_csv(__val_csv)
            csv_test = pd.read_csv(__test_csv)
            self.csv = csv_val.append(csv_test,ignore_index=True)
        else:
            csv_val = pd.read_csv(__val_csv)
            csv_test = pd.read_csv(__test_csv)
            self.csv = csv_val.append(csv_test, ignore_index=True)
            csv_train = pd.read_csv(__train_csv)
            self.csv = self.csv.append(csv_train, ignore_index=True)

        print('origin dataset len ',len(self.csv))
        class_name = self.csv.drop_duplicates(['label'])
        self.class_name = {}
        for i, name in enumerate(class_name['label']):
            if ten_class:
                if i >= 10:
                    break
            self.class_name.update({name: i})
        if ten_class:
            self.csv = self.csv[self.csv.apply(lambda x: x['label'] in self.class_name.keys(), axis=1)]

        self.csv = self.csv.to_numpy()

    @staticmethod
    def reconstruct_miniimagenet(csv, root):
        new_csv = os.path.join(root, "new-csv")
        print(new_csv)
        train = pd.DataFrame()
        test = pd.DataFrame()
        val = pd.DataFrame()
        class_name = csv.drop_duplicates(['label'])
        for name in class_name["label"]:
            temp = csv[csv.apply(lambda x: x['label'] is name, axis=1)]
            train = train.append(temp[:500], ignore_index=True)
            test  = test.append(temp[500:550], ignore_index=True)
            val   = val.append(temp[550:], ignore_index=True)
        train = train.reindex(np.random.permutation(train.index))
        test = test.reindex(np.random.permutation(test.index))
        val = val.reindex(np.random.permutation(val.index))
        train.to_csv(os.path.join(new_csv, 'train.csv'),index=False,header=True)
        test.to_csv(os.path.join(new_csv, 'test.csv'), index=False, header=True)
        val.to_csv(os.path.join(new_csv, 'val.csv'), index=False, header=True)

        print("reconstruct mini imagenet dataset ")

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        image, label = self.csv[index]
        label = self.class_name[label]
        image = Image.open(os.path.join(self.__img_dir, image))
        image = self.transform(image)
        return image, label


class miniImagenet(object):
    def __init__(self):
        self.root ="/media/kipp/work/Datas/mini-imagenet"
        self.num_work = 4
        self.shuffle =True

    def Transform(self, img_size):
        transform = [
            # transforms.RandomCrop(224),
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomAffine(5),
            transforms.Resize((img_size, img_size),Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]
        return transform

    def get_loader(self, batch_size, img_size, mode="test"):
        transform = self.Transform(img_size)
        return torch.utils.data.DataLoader(
            MiniImagenet(self.root,transform, train=mode),
            batch_size=batch_size,
            shuffle=self.shuffle,
            num_workers = self.num_work
        )
import matplotlib.pyplot as plt
def main():
    plt.figure()
    mini_imagenet = miniImagenet()
    loader = mini_imagenet.get_loader(1, 224)
    print(len(loader))
    for i,(images, labels) in enumerate(loader):
        if i >10:
            break
        #labels = F.one_hot(labels, num_classes=10)
        print(labels)
        dis_img = images[0].numpy().transpose(1,2,0)
        dis_img = (dis_img+1)/2
        plt.imshow(dis_img)
        plt.show()

if __name__ == "__main__":
    import fire
    fire.Fire(main)