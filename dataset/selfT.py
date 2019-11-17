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

def main():
    pass


if __name__ == "__main__":
    import fire

    fire.Fire(main)