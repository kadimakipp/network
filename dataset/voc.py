#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: voc.py
@time: 2019/11/13 下午5:25
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import collections
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import cv2
print(cv2.__file__)

import torchvision
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
#torchvision.datasets.VOCDetection()

"""
https://github.com/TencentYoutuResearch/ObjectDetection-OneStageDet/tree/master/yolo
https://github.com/chenyuntc/simple-faster-rcnn-pytorch/blob/master/data/voc_dataset.py
"""
class VOC_Auxiliary(object):
    def __init__(self):
        self.__classes__ = ('__background__', # always index 0
             'aeroplane', 'bicycle', 'bird', 'boat',
             'bottle', 'bus', 'car', 'cat', 'chair',
             'cow', 'diningtable', 'dog', 'horse',
             'motorbike', 'person', 'pottedplant',
             'sheep', 'sofa', 'train', 'tvmonitor')
        self.__name2int = dict(zip(self.__classes__, range(len(self.__classes__))))
        self.__int2name = dict(zip(range(len(self.__classes__)),self.__classes__))

    def name2ind(self, name):
        return self.__name2int[name]

    def ind2name(self, ind:int):
        return self.__int2name[ind]

    def bndbox2box(self, dict:dict):
        xmin = int(dict['xmin'])
        ymin = int(dict['ymin'])
        xmax = int(dict['xmax'])
        ymax = int(dict['ymax'])
        return [xmin, ymin, xmax, ymax]

    def parse(self, objects):
        if not isinstance(objects, list):
            objects = [objects]
        boxes = []
        clses = []
        for object in objects:
            c = self.name2ind(object['name'])
            b = self.bndbox2box(object['bndbox'])
            boxes.append(b)
            clses.append(c)
        return boxes, clses


class PascalVOC(Dataset):
    def __init__(self, root, transform,years='2007', mode='train',  segmentation=False):
        super(PascalVOC, self).__init__()
        self.root = root
        self.transform = transform
        self.years=years
        self.mode = mode
        self.segmentation=segmentation
        self.__aux = VOC_Auxiliary()
        if self.years not in ['2007', '2012']:
            raise RuntimeError("years must belong to ['2007', '2012']")
        if self.mode not in ["train", "trainval", "val"]:
            raise RuntimeError("mode must belong to ['train', 'trainval', 'val']")

        base_dir = 'VOC'+self.years
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')
        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.')
        splits_dir = os.path.join(voc_root, 'ImageSets/Main')
        split_f = os.path.join(splits_dir, self.mode.rstrip('\n')+'.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x+".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
       Args:
           index (int): Index

       Returns:
           tuple: (image, target) where target is a dictionary of the XML tree.
       """
        img = Image.open(self.images[index]).convert('RGB')
        # img = cv2.imread(self.images[index])
        anns = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        boxes_list, clses_list = self.__aux.parse(anns['annotation']['object'])
        boxes = np.zeros((32, 4))
        clses = np.zeros(32)
        for i, (box, cls) in enumerate(zip(boxes_list,clses_list)):
            boxes[i] = np.array(box)
            clses[i] = cls
        boxes = torch.from_numpy(boxes)
        clses = torch.from_numpy(clses)
        img = self.transform(img)
        return img, boxes, clses

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

class VOC(object):
    def __init__(self):
        self.root = "/media/kipp/work/Datas/VOCdevkit"
        self.num_work = 4
        self.shuffle = True

    def Transform(self, img_size):
        transform = transforms.Compose([
            # transforms.RandomCrop(224),
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomAffine(5),
            #transforms.Resize((img_size, img_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform

    def get_loader(self, batch_size, img_size, mode="all"):
        transform = self.Transform(img_size)
        return torch.utils.data.DataLoader(
            PascalVOC(self.root, transform),
            batch_size=batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_work
        )

import matplotlib.pyplot as plt
from samhi.colors import RGB_COLORS
# print(sys.path)
# if 'python2.7' in sys.path:
#     paths = [path for path in sys.path if 'python2.7' in path]
#     print(paths)
def img_writer(img, boxes, cls):
    dis_img = img.numpy().transpose(1,2,0)
    dis_img = (dis_img* [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])*255
    dis_img = dis_img.astype(np.uint8)
    dis_img = cv2.cvtColor(dis_img, cv2.COLOR_RGB2BGR)
    boxes = boxes.int().numpy()
    cls = cls.numpy().astype(np.uint8)
    for box, c in zip(boxes, cls):
        if c == 0:break
        cv2.rectangle(dis_img, (box[0], box[1]), (box[2],box[3]),RGB_COLORS[c], 2)
    # cv2.imshow("img",dis_img)
    # cv2.waitKey(0)
    return cv2.cvtColor(dis_img, cv2.COLOR_BGR2RGB)

def main():
    plt.figure()
    voc = VOC()
    loader = voc.get_loader(1, 224, "train")
    print(len(loader))
    for i, (images, boxes, cls) in enumerate(loader):
        print(images.shape, boxes.shape, cls.shape)
        dis_img = img_writer(images[0], boxes[0], cls[0])
        plt.imshow(dis_img)
        plt.show()
        break

if __name__ == "__main__":
    import fire

    fire.Fire(main)