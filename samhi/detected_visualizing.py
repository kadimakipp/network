#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: detected_visualizing.py
@time: 2019/11/20 上午10:45
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from samhi.colors import  RGB_COLORS as Colors
import cv2
import torchvision.transforms as transforms
from dataset import selfT
from dataset.coco import COCO,COCOAux
from samhi.detected import DetectedAux as DA
class YoloDataVis(object):
    def __init__(self):
        self.image_size = 416
        self.anchors = np.array([[116, 90], [156, 198], [373, 326],
                    [30, 61], [62, 45], [59, 119],
                    [10, 13], [16, 30], [33, 23]])
        self.feature_size = np.array([52,26,13])
        self.n_anchor = self.anchors.shape[0]
        self.n_classes = 81
        self.coco_aux = COCOAux()
        self.init_coco()

    def Transform(self, img_size):
        transform = [
            selfT.KeepAspect(),
            selfT.Resize(img_size),
            selfT.YoloTarget(self.n_classes, self.anchors,self.image_size,self.feature_size)
        ]
        return transforms.Compose(transform)
    def init_coco(self):
        root = '/media/kipp/work/DATASET/COCO'
        transform = self.Transform(self.image_size)
        self.coco = COCO(root, transform=transform,target_transform=transform,train='val', years='2017')

    def get_a_sample(self, image_id=102):
        return self.coco.__getitem__(image_id)

    def ann_visualization(self):
        sample = self.get_a_sample()
        image = sample['image']#numpy
        boxes = sample['bboxes']#numpy
        cls = sample['categories']#numpy
        print(image.shape, boxes.shape,cls.shape)
        h,w,_ = image.shape
        image=image.astype(np.uint8)

        for box, c in zip(boxes,cls):
            box = DA.cxywh2xywh(box)
            name = self.coco_aux.int2name(c)
            cv2.rectangle(image, (int(box[0] * w), int(box[1] * h)),
                      (int((box[0] + box[2]) * w), int((box[1] + box[3]) * h)), Colors[c], 1)
            cv2.putText(image, name, (int(box[0] * w), int(box[1] * h)), cv2.FONT_HERSHEY_SIMPLEX,
                        1, Colors[c], 2, cv2.LINE_AA)
        plt.figure()
        plt.imshow(image)
        plt.show()

    def target_visualization(self):
        sample = self.get_a_sample()
        image = sample['image']  # numpy
        boxes = sample['bboxes']  # numpy
        cls = sample['categories']  # numpy
        h, w, _ = image.shape
        # plot bbox
        for box, c in zip(boxes, cls):
            cv2.circle(image, (int(box[0] * w), int(box[1] * h)), 3, Colors[c], -1)
            box = DA.cxywh2xywh(box)
            cv2.rectangle(image, (int(box[0] * w), int(box[1] * h)),
                          (int((box[0] + box[2]) * w), int((box[1] + box[3]) * h)), Colors[c], 2)
        # plt.figure()
        # plt.imshow(image)
        # plt.show()
        for f_s in self.feature_size:
            key_str = 'scale_{}_'.format(f_s)
            # plot grid
            no_obj_mask = sample[key_str+'no_obj']#no_obj=1, obj=0
            obj_mask = sample[key_str+'obj']#obj=1 have object
            target = sample[key_str+'target']
            txs = target[0:3]
            tys = target[3:6]
            tws = target[6:9]
            ths = target[9:12]
            confs = target[12:15]
            cats = target[15:]
            print(obj_mask.shape, no_obj_mask.shape, target.shape)
            print(cats.shape)
            plt_img = image.copy()
            f = self.image_size//f_s
            for s in np.arange(0,self.image_size,f):
                cv2.line(plt_img, (s,0), (s,self.image_size), Colors[135], 1)#h
                cv2.line(plt_img, (0,s), (self.image_size,s), Colors[135], 1)#v
            #plot no obj grad
            no_obj_anchor,no_obj_x,no_obj_y = np.where(no_obj_mask==0)
            color=[Colors[125],Colors[128],Colors[133]]
            for (anchor_id, y, x) in zip(no_obj_anchor,no_obj_x,no_obj_y):
                cv2.circle(plt_img, (x*f, y*f), 5, color[anchor_id], 2)

            #plot obj grad
            obj_anchor, obj_x, obj_y = np.where(obj_mask == 1)
            color = [Colors[125],Colors[128],Colors[133]]
            for (anchor_id, y, x) in zip(obj_anchor, obj_x, obj_y):
                p_start = (x*f, y*f)
                cv2.circle(plt_img, p_start, 3, color[anchor_id], -1)
                tx = txs[anchor_id,y,x]
                ty = tys[anchor_id,y,x]
                p_end = (x*f+int(tx*f), y*f+int(ty*f))
                cv2.line(plt_img, p_start,p_end, color[anchor_id], 2)
                assert confs[anchor_id, y, x] == 1
                th = ths[anchor_id, y, x]
                tw = tws[anchor_id, y, x]
                #TODO: visualization th, tw
                print(th,tw)

            plt.figure()
            plt.imshow(plt_img)
            plt.show()

def main():
    vis = YoloDataVis()
    # vis.ann_visualization()
    vis.target_visualization()

if __name__ == "__main__":
    import fire

    fire.Fire(main)