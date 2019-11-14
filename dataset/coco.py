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
import cv2
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pycocotools.coco import COCO as COCOtool
"""
https://github.com/BobLiu20/YOLOv3_PyTorch/blob/master/common/coco_dataset.py
"""
class COCOAux(object):
    def __init__(self):
        self.__classes__ = ('__background__', # always index 0
         'person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant',
         'stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra',
         'giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
         'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup',
         'fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
         'donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse',
         'remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase',
         'scissors','teddy bear','hair drier','toothbrush')
        self.__name2int = dict(zip(self.__classes__, range(len(self.__classes__))))
        self.__int2name = dict(zip(range(len(self.__classes__)), self.__classes__))

        self.categroies = [{'name': 'person', 'id': 1, 'supercategory': 'person'}, {'name': 'bicycle', 'id': 2, 'supercategory': 'vehicle'},
                           {'name': 'car', 'id': 3, 'supercategory': 'vehicle'}, {'name': 'motorcycle', 'id': 4, 'supercategory': 'vehicle'},
                           {'name': 'airplane', 'id': 5, 'supercategory': 'vehicle'}, {'name': 'bus', 'id': 6, 'supercategory': 'vehicle'},
                           {'name': 'train', 'id': 7, 'supercategory': 'vehicle'}, {'name': 'truck', 'id': 8, 'supercategory': 'vehicle'},
                           {'name': 'boat', 'id': 9, 'supercategory': 'vehicle'}, {'name': 'traffic light', 'id': 10, 'supercategory': 'outdoor'},
                           {'name': 'fire hydrant', 'id': 11, 'supercategory': 'outdoor'}, {'name': 'stop sign', 'id': 13, 'supercategory': 'outdoor'},
                           {'name': 'parking meter', 'id': 14, 'supercategory': 'outdoor'}, {'name': 'bench', 'id': 15, 'supercategory': 'outdoor'},
                           {'name': 'bird', 'id': 16, 'supercategory': 'animal'}, {'name': 'cat', 'id': 17, 'supercategory': 'animal'},
                           {'name': 'dog', 'id': 18, 'supercategory': 'animal'}, {'name': 'horse', 'id': 19, 'supercategory': 'animal'},
                           {'name': 'sheep', 'id': 20, 'supercategory': 'animal'}, {'name': 'cow', 'id': 21, 'supercategory': 'animal'},
                           {'name': 'elephant', 'id': 22, 'supercategory': 'animal'}, {'name': 'bear', 'id': 23, 'supercategory': 'animal'},
                           {'name': 'zebra', 'id': 24, 'supercategory': 'animal'}, {'name': 'giraffe', 'id': 25, 'supercategory': 'animal'},
                           {'name': 'backpack', 'id': 27, 'supercategory': 'accessory'}, {'name': 'umbrella', 'id': 28, 'supercategory': 'accessory'},
                           {'name': 'handbag', 'id': 31, 'supercategory': 'accessory'}, {'name': 'tie', 'id': 32, 'supercategory': 'accessory'},
                           {'name': 'suitcase', 'id': 33, 'supercategory': 'accessory'}, {'name': 'frisbee', 'id': 34, 'supercategory': 'sports'},
                           {'name': 'skis', 'id': 35, 'supercategory': 'sports'}, {'name': 'snowboard', 'id': 36, 'supercategory': 'sports'},
                           {'name': 'sports ball', 'id': 37, 'supercategory': 'sports'}, {'name': 'kite', 'id': 38, 'supercategory': 'sports'},
                           {'name': 'baseball bat', 'id': 39, 'supercategory': 'sports'}, {'name': 'baseball glove', 'id': 40, 'supercategory': 'sports'},
                           {'name': 'skateboard', 'id': 41, 'supercategory': 'sports'}, {'name': 'surfboard', 'id': 42, 'supercategory': 'sports'},
                           {'name': 'tennis racket', 'id': 43, 'supercategory': 'sports'}, {'name': 'bottle', 'id': 44, 'supercategory': 'kitchen'},
                           {'name': 'wine glass', 'id': 46, 'supercategory': 'kitchen'}, {'name': 'cup', 'id': 47, 'supercategory': 'kitchen'},
                           {'name': 'fork', 'id': 48, 'supercategory': 'kitchen'}, {'name': 'knife', 'id': 49, 'supercategory': 'kitchen'},
                           {'name': 'spoon', 'id': 50, 'supercategory': 'kitchen'}, {'name': 'bowl', 'id': 51, 'supercategory': 'kitchen'},
                           {'name': 'banana', 'id': 52, 'supercategory': 'food'}, {'name': 'apple', 'id': 53, 'supercategory': 'food'},
                           {'name': 'sandwich', 'id': 54, 'supercategory': 'food'}, {'name': 'orange', 'id': 55, 'supercategory': 'food'},
                           {'name': 'broccoli', 'id': 56, 'supercategory': 'food'}, {'name': 'carrot', 'id': 57, 'supercategory': 'food'},
                           {'name': 'hot dog', 'id': 58, 'supercategory': 'food'}, {'name': 'pizza', 'id': 59, 'supercategory': 'food'},
                           {'name': 'donut', 'id': 60, 'supercategory': 'food'}, {'name': 'cake', 'id': 61, 'supercategory': 'food'},
                           {'name': 'chair', 'id': 62, 'supercategory': 'furniture'}, {'name': 'couch', 'id': 63, 'supercategory': 'furniture'},
                           {'name': 'potted plant', 'id': 64, 'supercategory': 'furniture'}, {'name': 'bed', 'id': 65, 'supercategory': 'furniture'},
                           {'name': 'dining table', 'id': 67, 'supercategory': 'furniture'}, {'name': 'toilet', 'id': 70, 'supercategory': 'furniture'},
                           {'name': 'tv', 'id': 72, 'supercategory': 'electronic'}, {'name': 'laptop', 'id': 73, 'supercategory': 'electronic'},
                           {'name': 'mouse', 'id': 74, 'supercategory': 'electronic'}, {'name': 'remote', 'id': 75, 'supercategory': 'electronic'},
                           {'name': 'keyboard', 'id': 76, 'supercategory': 'electronic'}, {'name': 'cell phone', 'id': 77, 'supercategory': 'electronic'},
                           {'name': 'microwave', 'id': 78, 'supercategory': 'appliance'}, {'name': 'oven', 'id': 79, 'supercategory': 'appliance'},
                           {'name': 'toaster', 'id': 80, 'supercategory': 'appliance'}, {'name': 'sink', 'id': 81, 'supercategory': 'appliance'},
                           {'name': 'refrigerator', 'id': 82, 'supercategory': 'appliance'}, {'name': 'book', 'id': 84, 'supercategory': 'indoor'},
                           {'name': 'clock', 'id': 85, 'supercategory': 'indoor'}, {'name': 'vase', 'id': 86, 'supercategory': 'indoor'},
                           {'name': 'scissors', 'id': 87, 'supercategory': 'indoor'}, {'name': 'teddy bear', 'id': 88, 'supercategory': 'indoor'},
                           {'name': 'hair drier', 'id': 89, 'supercategory': 'indoor'}, {'name': 'toothbrush', 'id': 90, 'supercategory': 'indoor'}]


    def name2ind(self, name):
        return self.__name2int[name]

    def ind2name(self, ind:int):
        return self.__int2name[ind]

    def xywh2xyxy(self, bbox):
        assert isinstance(bbox, list)
        return [bbox[0], bbox[1], bbox[0]+bbox[2],bbox[1]+bbox[3]]

    def parse_coco_ann(self, objects):
        """box: x y w h"""
        assert isinstance(objects, list)
        categories = []
        bboxes = []
        #iscrowds = []
        #names = []
        for o in objects:
            #iscrowds.append(o['iscrowd'])
            #names.append(o['image_id'])
            categories.append(o['category_id'])
            bboxes.append(self.xywh2xyxy(o['bbox']))

        return categories,bboxes

    @classmethod
    def image_dir(cls, mode='val', years='2014'):
        assert mode in ['train','val', 'test']
        assert years in ['2014', '2017']
        return mode+years

#TODO: train and val 2014 2017
class COCO(Dataset):
    def __init__(self, root, transform, target_transform,train='train', years='2014'):
        super(COCO, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.annFile = os.path.join(self.root, 'annotations2014/instances_minival2014.json')

        self.aux = COCOAux()
        self.max_object = 64
        self.img_dir = os.path.join(self.root, COCOAux.image_dir('val',years))

        self.coco = COCOtool(self.annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.categories = self.coco.dataset['categories']

    def __getitem__(self, index):
        """
        :param index:
        :return:Tuple(image, target)
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        file_name = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_dir, file_name)).convert('RGB')
        categoriesL,bboxesL = self.aux.parse_coco_ann(target)
        bboxes = np.zeros((self.max_object,4))
        clses = np.zeros(self.max_object)
        for i, (bbox,cls) in enumerate(zip(bboxesL,categoriesL)):
            bboxes[i] = np.array(bbox)
            clses[i] = cls
        img = self.transform(img)
        bboxes = torch.from_numpy(bboxes)
        clses = torch.from_numpy(clses)

        return img, bboxes, clses

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
            # transforms.Resize((img_size, img_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        return transforms.Compose(transform)


    def get_loader(self, batch_size, img_size, mode='train'):
        transform= self.Transform(img_size)
        return torch.utils.data.DataLoader(
            COCO(self.root, transform=transform,target_transform=transform,train=mode),
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
    coco = CoCo()
    loader = coco.get_loader(1,224, 'train')
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