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

        total_loss = 0
        #three scale
        for level_key in self.level_keys:
            pre = out[level_key]
            gt = target[level_key]
            n_class = gt.shape[1] // 3 - 7
            pre_tuple = pre.split([3,3,3,3,3,n_class*3],dim=1)
            gt_tuple = gt.split([3,3,3,3,3,3,3,n_class*3], dim=1)
            # g---- ground truth
            no_obj_mask,obj_mask,gx,gy,gw,gh,gconf,gcla = gt_tuple
            #p --predicted
            px,py,pw,ph,pconf,pcla = pre_tuple
            px = self.sigmoid(px)
            py = self.sigmoid(py)
            pconf = self.sigmoid(pconf)
            pcla = self.sigmoid(pcla)

            loss_x = self.mse(px * obj_mask, gx * obj_mask)
            loss_y = self.mse(py * obj_mask, gy * obj_mask)
            loss_w = self.mse(pw * obj_mask, gw * obj_mask)
            loss_h = self.mse(ph * obj_mask, gh * obj_mask)

            loss_conf = self.bce(pconf*obj_mask, gconf)+\
                0.5*self.bce(pconf*no_obj_mask, gconf*no_obj_mask)
            obj_mask = torch.repeat_interleave(obj_mask,n_class,dim=1)
            loss_cls = self.bce(pcla[obj_mask==1], gcla[obj_mask==1])

            loss_detail[level_key] = [self.lambda_loc*(loss_x+loss_y+loss_w+loss_h),
                                      self.lambda_conf*loss_conf,
                                      self.lambda_cls*loss_cls]

            total_loss += self.lambda_loc*(loss_x+loss_y+loss_w+loss_h)+\
                   self.lambda_conf*loss_conf+\
                   self.lambda_cls*loss_cls

        return total_loss, loss_detail


from samhi.detected import DetectedAux as aux
class YoloInference(object):
    def __init__(self, anchors, n_classes, feature_size,image_size,
                 obj_threshold,score_threshold,nms_threshold):
        super(YoloInference, self).__init__()
        self.obj_threshold = obj_threshold
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        self.image_size = image_size
        self.anchors = anchors


        self.n_anchors = anchors.shape[0]
        self.n_classes = n_classes
        self.feature_size = feature_size #[52,26,13]
        self.level_keys = ['one', 'two', 'three']
        self.sigmoid = nn.Sigmoid()
        self.anchors_tuple = torch.split(torch.from_numpy(self.anchors),3,dim=0)

    def forward(self, out):
        if out[self.level_keys[0]].is_cuda:
            self.anchors_tuple = [an.cuda() for an in self.anchors_tuple]
        n_anchor = self.n_anchors//len(self.level_keys)
        level_det = {}
        for level_key, anchors in zip(self.level_keys,self.anchors_tuple):
            pre = out[level_key].detach()
            bn, _, f_h, f_w = pre.shape
            pre_tuple = pre.split([n_anchor + n_anchor,
                                   n_anchor + n_anchor,
                                   n_anchor,
                                   self.n_classes * n_anchor], dim=1)
            pxy, pwh, pconf, pcla = pre_tuple
            # pxy = self.sigmoid(pxy)
            # pconf = self.sigmoid(pconf)
            # pcla = self.sigmoid(pcla)
            # (bn, n_class*3, f_h, f_w) reshape (bn, 3, n_class, f_h,f_w)
            pcla_t = pcla.split(self.n_classes, dim=1)
            pclaL = [pc.unsqueeze_(dim=1) for pc in pcla_t]
            pcla = torch.cat(pclaL, dim=1)
            #get score categories
            pscore, pcat = torch.max(pcla,dim=2,keepdim=True)
            pcat.squeeze_(dim=2)
            pscore.squeeze_(dim=2)
            #(bn, 3  f_h, f_w)
            px,py = pxy.split([n_anchor,n_anchor],dim=1)
            #(bn, 3, f_h, f_w)
            grid_x = torch.arange(0, f_w).view(1,-1)#(1,f_w)
            grid_y = torch.arange(0, f_h).view(-1,1)#(f_h,1)
            px = (px+grid_x)/f_w
            py = (py+grid_y)/f_h
            pw,ph = pwh.split([n_anchor,n_anchor],dim=1)
            an_w, an_h = anchors[:,0], anchors[:,1]#(3)
            an_w = an_w.view(n_anchor,1,1).float()
            an_h = an_h.view(n_anchor,1,1).float()
            #TODO: params
            pw = torch.exp(pw)*an_w/self.image_size
            ph = torch.exp(ph)*an_h/self.image_size
            x1 = px - pw / 2.
            y1 = py - ph / 2.
            x2 = px + pw / 2.
            y2 = py + ph / 2.
            x1[x1 < 0] = 0.
            y1[y1 < 0] = 0.
            x2[x2 > 1] = 1.
            y2[y2 > 1] = 1.
            masks = pconf>self.obj_threshold
            pcat = pcat.double()#data type transform

            bn_det = []
            #px,py,pw,ph,pcat,pscore
            for i, (m,x_min,y_min,x_max,y_max,cat,score) in \
                    enumerate(zip(masks, x1, y1, x2, y2, pcat, pscore)):#each image
                detL = [x_min,y_min,x_max,y_max, cat, score]
                detL = [d[m].view(-1,1) for d in detL]
                det = torch.cat(detL,dim=1)#(Nboxes,6)
                #score threshold
                if self.score_threshold>0:
                    score_mask = det[:,5]>self.score_threshold
                    det = det[score_mask]
                bn_det.append(det)
            level_det[level_key] = bn_det # batch_size, (Nboxes, 6)

        #level merge
        bn_det = []
        bn = out[self.level_keys[0]].shape[0]
        for n in range(bn):
            batch = []
            for k in self.level_keys:
                batch.append(level_det[k][n])
            batch_det = torch.cat(batch, dim=0)
            batch_det = self.NMS(batch_det, self.nms_threshold)
            bn_det.append(batch_det)
        return bn_det



    @staticmethod
    def NMS(det, nms_threshold):
        """

        :param det: [x1, y1, x2, y2, cls, score](N, 6) tensor
        :param nms_threshold:
        :return:
        """
        #NMS
        ## sort socre
        score_indices = torch.argsort(det[:,5],descending=True)
        detection = det[score_indices]
        cats = torch.unique(detection[:,4])
        detections = []
        for c in cats:#each class
            cat_mask = detection[:,4]==c
            det = detection[cat_mask]
            while det.shape[0]:
                max_det = det[0].unsqueeze(dim=0)
                detections.append(max_det)
                if det.shape[0] == 1:
                    break
                #Get the IOUs for all boxes with lower confidence
                iou = aux.iou(max_det,det[1:])
                #remove iou>self.nms_threshold
                det = det[1:][iou<=nms_threshold]

        return torch.cat(detections, dim=0)



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