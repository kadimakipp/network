#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: detected.py
@time: 19-11-18 下午11:19
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

class DetectedAux(object):
    def __init__(self):
        pass

    @staticmethod
    def xywh2xyxy(bbox):
        assert isinstance(bbox, list)
        return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

    @staticmethod
    def xywh2cxywh(bbox):
        if isinstance(bbox, list):  # 1D
            bb = [float(b) for b in bbox]
            bbox = [bb[0] + bb[2] / 2, bb[1] + bb[3] / 2, bb[2], bb[3]]
        elif isinstance(bbox, np.ndarray):  # 2D
            bbox = bbox.astype(np.float)
            if len(bbox.shape) == 1:
                bbox[0] = bbox[0] + bbox[2] / 2
                bbox[1] = bbox[1] + bbox[3] / 2
            elif len(bbox.shape) == 2:
                bbox[:, 0] = bbox[:, 0] + bbox[:, 2] / 2
                bbox[:, 1] = bbox[:, 1] + bbox[:, 3] / 2
        else:
            raise Exception("xywh to cxcywh error")
        return bbox

    @staticmethod
    def cxywh2xywh(bbox):
        if isinstance(bbox, list):  # 1D
            bbox = [bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2, bbox[2], bbox[3]]
        elif isinstance(bbox, np.ndarray):  # 2D
            if len(bbox.shape) == 1:
                bbox[0] = bbox[0] - bbox[2] / 2
                bbox[1] = bbox[1] - bbox[3] / 2
            elif len(bbox.shape) == 2:
                bbox[:, 0] = bbox[:, 0] - bbox[:, 2] / 2
                bbox[:, 1] = bbox[:, 1] - bbox[:, 3] / 2
        else:
            raise Exception("cxcywh to xywh error")
        return bbox

    @staticmethod
    def iou(a_box, b_box):
        """
        must be same shape,one to one,you can repeat a_box,b_box
        :param a_box:x y x y ,shape(N,4)
        :param b_box:x y x y ,shape(N,4)
        :return:
        """
        assert a_box.shape == b_box.shape
        Fmax = np.maximum
        Fmin = np.minimum
        if torch.is_tensor(a_box) and torch.is_tensor(b_box):
            Fmax = torch.max
            Fmin = torch.min
        ax1,ay1,ax2,ay2 = a_box[:,0], a_box[:,1], a_box[:,2], a_box[:,3]
        bx1,by1,bx2,by2 = b_box[:,0], b_box[:,1], b_box[:,2], b_box[:,3]
        #inter rect
        x1 = Fmax(ax1,bx1)
        y1 = Fmax(ay1,by1)
        x2 = Fmin(ax2,bx2)
        y2 = Fmin(ay2,by2)
        inter_w = x2-x1
        inter_h = y2-y1
        inter_w[inter_w<0] = 0
        inter_h[inter_h<0] = 0
        inter_rect_area = inter_w*inter_h

        a_area = (ax2-ax1)*(ay2-ay1)
        b_area = (bx2-bx1)*(by2-by1)
        #inter/union
        iou = inter_rect_area/(a_area+b_area-inter_rect_area+1e-16)
        return iou

def test_iou():
    print("-------------one to one------------")
    # a1----b1
    # a2----b2
    # a3----b3
    a_box = np.array([[0.05, 0.1, 0.2, 0.3],
                      [0.5, 0.5, 0.8, 0.8],
                      [0.3, 0.2, 0.6, 0.8]])
    b_box = np.array([[0.3, 0.4, 0.5, 0.6],
                      [0.6, 0.6, 0.7, 0.9],
                      [0.3, 0.2, 0.6, 0.8]])
    aux = DetectedAux()
    iou = aux.iou(a_box, b_box)
    print(iou)
    a_box_t = torch.from_numpy(a_box)
    b_box_t = torch.from_numpy(b_box)
    iou_t = aux.iou(a_box_t, b_box_t)
    print(iou_t)
    # a1-----b1
    # a1-----b2
    #   ...
    # a2-----b1
    # a2-----b2
    #   ...
    print("---------one to all------------")
    a_boxs = a_box.repeat(b_box.shape[0], axis=0)
    print(a_boxs)

    b_boxs = np.tile(b_box, (a_box.shape[0], 1))
    print(b_boxs)
    iou = aux.iou(a_boxs, b_boxs)
    print(iou)
    a_boxs_t = torch.from_numpy(a_boxs)
    b_boxs_t = torch.from_numpy(b_boxs)
    iou_t = aux.iou(a_boxs_t, b_boxs_t)
    print(iou_t)

def main():
    test_iou()

if __name__ == "__main__":
    import fire

    fire.Fire(main)