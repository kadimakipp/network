#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: selfT.py
@time: 19-11-17 下午4:55
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torchvision
import cv2

"""
custom Transform for pytorch
"""

class KeepAspect(object):
    """
    keep image and label box is keep square
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        image = sample["image"]
        bboxes = sample["bboxes"]
        h,w,_ = image.shape

        dim_diff = np.abs(h-w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))

        image_new = np.pad(image, pad, 'constant', constant_values=128)
        padded_h, padded_w, _ = image_new.shape
        #padd bboxes
        bboxes[:,0] += pad[1][0]
        bboxes[:,1] += pad[0][0]
        # Calculate ratios from coordinates
        bboxes[:,[0,2]]/=padded_w
        bboxes[:,[1,3]]/=padded_h
        sample['image'] = image_new
        sample['bboxes'] = bboxes

        return sample

class Resize(object):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        """
        Size is int or tuple
        """
        self.size = size#tuple(w,h)
        if not isinstance(self.size, tuple):
            self.size = (self.size, self.size)
        self.interpolation = interpolation

    def __call__(self, sample):
        #resize image
        image= sample['image']
        image = cv2.resize(image, self.size, interpolation=self.interpolation)
        sample['image'] = image
        return sample

class ToTensor(object):
    def __init__(self, max_object):
        self.max_object = max_object

    def __call__(self, sample):
        image, bboxes, cls = sample['image'], sample['bboxes'],sample['categories']
        image = np.array(image).transpose((2, 0, 1)).astype(np.float32)/255.0
        image = torch.from_numpy(image)
        b = np.zeros((self.max_object, 4))
        c = np.zeros(self.max_object)
        b[range(bboxes.shape[0])[:self.max_object]] = bboxes[:self.max_object]
        c[range(bboxes.shape[0])[:self.max_object]] = cls[:self.max_object]
        bboxes = torch.from_numpy(b)
        cls = torch.from_numpy(c)
        sample['image'] = image
        sample['bboxes'] = bboxes
        sample['categories'] = cls

        return sample

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=True):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        torchvision.transforms.functional.normalize(sample['image'], self.mean, self.std, self.inplace)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


from samhi.detected import DetectedAux as aux
class YoloTarget(object):
    """
    build yolo network target
    for coco example
    input:  classes, 80
            anchors [[116, 90], [156, 198], [373, 326],
                    [30, 61], [62, 45], [59, 119],
                    [10, 13], [16, 30], [33, 23]] narray shape(9,2)
            size = 416 or (416,416)
            ignore_threshold=0.5 iou>ignore_threshold, Think of the anchor have object

    output: [mask,noobj_mask,tx,ty,tw,th,confidence, classes'],
            shape(7+classes, anchors, h, w)
            classes is one hot encode
            mask = positive sample
            no obj mask = Negative sample
    """
    def __init__(self,classes, anchors, size, ignore_threshold):
        self.size = size  # tuple(w,h)
        if not isinstance(self.size, tuple):
            self.size = (self.size, self.size)
        self.h,self.w = self.size

        self.n_classes = classes
        self.anchors = anchors
        self.n_anchors = self.anchors.shape[0]
        self.ignore_threshold = ignore_threshold

    def __call__(self, sample):
        #TODO:Quesion : grid is feature or image; --image
        #anchor box原本设定是相对于416*416坐标系下的坐标，在yolov3.cfg文件中写明了
        # ，代码中是把cfg中读取的坐标除以stride如32映射到feature map坐标系中


        boxes = sample['bboxes']  # cxywh Non=0
        cls = sample['categories']
        cx=boxes[:,0]
        cy=boxes[:,1]
        w=boxes[:,2]
        h=boxes[:,3]
        # box top left corner translate to (0,0)
        #left_boxes = (0,0,w,h)
        Nboxes= boxes.shape[0]
        left_w = w[:, np.newaxis]
        left_h = h[:, np.newaxis]
        left_boxes = np.concatenate((np.zeros((Nboxes,2)),left_w,left_h), axis=1)
        anchor_boxes = np.concatenate((np.zeros(self.n_anchors,2), self.anchors), axis=1)

        left_boxes = left_boxes.repeat(self.n_anchors,axis=0)
        anchor_boxes=np.tile(anchor_boxes,(Nboxes,1))
        anchor_iou = aux.iou(left_boxes,anchor_boxes)
        anchor_level_n = int(np.sqrt(self.n_anchors))
        anchor_iou = anchor_iou.reshape(-1, anchor_level_n, anchor_level_n)#(Nboxes,3,3)

        #no obj
        anchor_mask = anchor_iou>self.ignore_threshold
        #mask map to no_obj
        #obj
        anchor_mask = np.argmax(anchor_iou,axis=2)
        #-----------------------------------------------------
        # a = x*feature   预测时与之对应即可。

        #3个尺度相互呼应。
        # init
        shape = (anchor_level_n,anchor_level_n, self.h, self.w)
        obj = np.zeros(shape)
        no_obj = np.ones(shape)
        tx = np.zeros(shape, dtype=np.float32)  #
        ty = np.zeros(shape, dtype=np.float32)  # 需要分尺度处理。
        tw = np.zeros(shape, dtype=np.float32)
        th = np.zeros(shape, dtype=np.float32)
        confidence = np.zeros(shape)
        categories = np.zeros(shape + (self.n_classes,))





        sample['target'] = None

        return sample


def main():
    pass


if __name__ == "__main__":
    import fire

    fire.Fire(main)