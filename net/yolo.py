#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: yolo.py
@time: 2019/11/19 下午7:10
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from net.darkent import DarkNet as backbone
"""
you only look once
V3
"""
# like darknet ResidualConv but some difference
class ConvOne(nn.Sequential):
    def __init__(self, inc, ouc):
        super(ConvOne, self).__init__(
            nn.Conv2d(in_channels=inc,out_channels=ouc,kernel_size=1),
            nn.BatchNorm2d(ouc),
            nn.LeakyReLU(inplace=True),
        )

class ConvThree(nn.Sequential):
    def __init__(self, inc,ouc):
        super(ConvThree, self).__init__(
            nn.Conv2d(in_channels=inc,out_channels=ouc,kernel_size=3,padding=1),
            nn.BatchNorm2d(ouc),
            nn.LeakyReLU(inplace=True)
        )

class ConvEnd(nn.Sequential):
    def __init__(self,inc, classes):
        ouc = 3*(4+1+classes)
        super(ConvEnd, self).__init__(
            nn.Conv2d(in_channels=inc,out_channels=ouc,kernel_size=1)
        )

class ConvPred(nn.Module):
    def __init__(self, inc, mic, ouc, classes):
        super(ConvPred, self).__init__()
        self.input = ConvOne(inc, mic)
        self.one = ConvOne(ouc,mic)
        self.three = ConvThree(mic,ouc)
        self.end = ConvEnd(ouc, classes)

    def forward(self, x):
        o = self.input(x)
        o = self.three(o)
        o = self.one(o)
        o = self.three(o)
        branch = self.one(o)

        o = self.three(branch)
        end = self.end(o)
        return end, branch

class UpSample(nn.Module):
    def __init__(self, inc):
        super(UpSample, self).__init__()
        ouc = inc//2
        self.conv = ConvOne(inc, ouc)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, stage, last_stage):
        o = self.conv(stage)
        o = self.up(o)
        out = torch.cat((o, last_stage),dim=1)
        return out

class YOLO(nn.Module):
    def __init__(self, inc, classes):
        super(YOLO, self).__init__()
        self.backbone = backbone(inc)

        self.branch_three = ConvPred(inc=1024,mic=512,ouc=1024, classes=classes)
        self.link_branch_three_two=UpSample(512)

        self.branch_two = ConvPred(inc=256+512, mic=256,ouc=512, classes=classes)
        self.link_branch_two_one = UpSample(256)

        self.branch_one = ConvPred(inc=128+256, mic=128, ouc=256, classes=classes)

    def forward(self, input):
        backbone = self.backbone(input)
        one, two, three = backbone['one'], backbone['two'], backbone['three']

        #13x13
        three, three_branch = self.branch_three(three)
        two = self.link_branch_three_two(three_branch, two)

        #26x26
        two, two_branch = self.branch_two(two)
        one = self.link_branch_two_one(two_branch, one)

        #52x52
        one, _ = self.branch_one(one)

        out = {'one': one, 'two': two, 'three': three}
        return out

class YOLO_Loss(nn.Module):
    def __init__(self):
        super(YOLO_Loss, self).__init__()
        self.lambda_loc=2.5
        self.lambda_conf = 1.
        self.lambda_cls = 1.
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCELoss(reduction='sum') # before need sigmoid
        self.level_keys = ['one','two', 'three']
        self.sigmoid = nn.Sigmoid()


    def forward(self, out, target):
        loss_detail = {'one':0, 'two':0, 'three':0}
        n_class = target.shape[1]//3-7
        total_loss = 0
        #three scale
        for level_key in self.level_keys:
            pre = out[level_key]
            gt = target[level_key]
            pre_tuple = pre.split([3+3,3+3,3,n_class*3],dim=1)
            gt_tuple = gt.split([3,3,3+3,3+3,3,n_class*3], dim=1)
            # g---- ground truth
            no_obj_mask,obj_mask,gxy,gwh,gconf,gcla = gt_tuple
            #p --predicted
            pxy,pwh,pconf,pcla = pre_tuple
            pxy = self.sigmoid(pxy)
            pconf = self.sigmoid(pconf)
            pcla = self.sigmoid(pcla)

            loss_xy = self.mse(pxy*obj_mask, gxy*obj_mask)
            loss_wh = self.mse(pwh*obj_mask, gwh*obj_mask)

            loss_conf = self.bce(pconf*obj_mask, gconf)+\
                0.5*self.bce(pconf*no_obj_mask, gconf*no_obj_mask)
            loss_cls = self.bce(pcla[obj_mask==1], gcla[obj_mask==1])

            loss_detail[level_key] = [self.lambda_loc*(loss_xy+loss_wh),
                                      self.lambda_conf*loss_conf,
                                      self.lambda_cls*loss_cls]

            total_loss += self.lambda_loc*(loss_xy+loss_wh)+\
                   self.lambda_conf*loss_conf+\
                   self.lambda_cls*loss_cls

        return total_loss, loss_detail


class YoloInference(object):
    def __init__(self, anchors, n_classes, feature_size):
        super(YoloInference, self).__init__()
        self.anchors = anchors
        self.n_anchors = anchors.shape[0]
        self.n_classes = n_classes
        self.feature_size = feature_size #[52,26,13]
        self.level_keys = ['one', 'two', 'three']
        self.sigmoid = nn.Sigmoid()
        self.anchors_tuple = torch.split(torch.from_numpy(self.anchors),3,dim=0)

    def forward(self, out):
        if out.is_cuda():
            self.anchors_tuple = [an.cuda() for an in self.anchors_tuple]
        n_anchor = self.n_anchors//3
        for level_key, anchors in zip(self.level_keys,self.anchors_tuple):
            pre = out[level_key]
            _, _, f_h, f_w = pre.shape
            pre_tuple = pre.split([3 + 3, 3 + 3, 3, self.n_classes * 3], dim=1)
            pxy, pwh, pconf, pcla = pre_tuple
            pxy = self.sigmoid(pxy)
            pconf = self.sigmoid(pconf)
            pcla = self.sigmoid(pcla)


            #TODO:reconstuctor code...., becase willing do what ??
            #(batch_size, 3,3 f_h,f_w)
            px,py = pxy.split([3,3],dim=1)

            pw,ph = pwh.split([3,3],dim=1)
            an_w,an_h = anchors.split(2,dim=1)
            pw = torch.exp(pw)















def main():
    input = torch.randn(4,3,416,416)
    yolo = YOLO(3, 80)
    print(yolo)
    out = yolo(input)
    for k in ['one','two', 'three']:
        print(k, out[k].shape)



if __name__ == "__main__":
    import fire
    fire.Fire(main)