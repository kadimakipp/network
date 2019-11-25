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
        self.anchors = np.array([[373, 326],[156, 198],[116, 90],
                    [59, 119],[62, 45],[30, 61],
                    [33, 23], [16, 30],[10, 13]])
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
        print(self.coco.__len__())
        #print(self.coco.ids[4779])#384,4169,1988

    def get_a_sample(self, image_id=20):#1072
        if image_id is None:
            image_id = np.random.randint(self.coco.__len__())
            print('image_id = ',image_id)
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
        target_keys = ['one','two', 'three']
        anchors = self.anchors.reshape((3,3,2))
        for f_s, ancs,k in zip(self.feature_size, anchors, target_keys):
            # plot grid
            target = sample[k]
            no_obj_mask = target[0:3]
            obj_mask = target[3:6]
            txs = target[6:9]
            tys = target[9:12]
            tws = target[12:15]
            ths = target[15:18]
            confs = target[18:21]
            cats = target[21:]
            print(obj_mask.shape, no_obj_mask.shape, target.shape)
            print('categories shape: ',cats.shape)
            #reshape cats to (3, n_classes, f_s,f_s)
            catsL = np.split(cats,3,axis=0)
            catsL = [l[np.newaxis,:] for l in catsL]
            cats = np.concatenate(catsL,axis=0)
            print("categories reshape: ",cats.shape)

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
            obj_anchor, obj_y, obj_x = np.where(obj_mask == 1)
            color = [Colors[125],Colors[128],Colors[133]]
            print(obj_x.shape)
            for (anchor_id, y, x) in zip(obj_anchor, obj_y, obj_x):
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
                org_h = np.exp(th)*ancs[anchor_id,1]
                org_w = np.exp(tw)*ancs[anchor_id,0]
                p_start = (int((x+tx)*f-org_w/2), int((y+ty)*f-org_h/2))
                p_end = (int((x+tx)*f+org_w/2), int((y+ty)*f+org_h/2))
                cv2.line(plt_img, p_start, p_end, color[anchor_id],1)
                cat = cats[anchor_id,:,y,x]
                cat = np.where(cat==1)[0][0]
                name = self.coco_aux.int2name(cat)
                cv2.putText(plt_img, name, (int((x+tx)*f), int((y+ty)*f)), cv2.FONT_HERSHEY_SIMPLEX,
                            1, Colors[cat], 1, cv2.LINE_AA)
            plt.figure()
            plt.imshow(plt_img)
            plt.show()


from net.yolo import YOLO_Loss, YoloInference
from dataset.coco import CoCo

class YOLOTest(object):
    def __init__(self):

        coco = CoCo()
        self.loader = coco.get_loader(2, 416)
        # -------loss-------
        self.level_keys = ['one', 'two', 'three']
        print("Attention!!! run this program need remove Sigmoid function")

    def loss_test(self):
        self.loss = YOLO_Loss()
        for i, samples in enumerate(self.loader):
            out = {}
            for k in self.level_keys:
                gt = samples[k]
                o_s = gt.shape[1] - 6  # (no obj,obj)
                _, _, out[k] = gt.split([3, 3, (o_s)], dim=1)
            loss = self.loss(out, samples)
            print(loss)
            break

    @staticmethod
    def img_writer(img, boxes, cls, target=True):
        """

        target = True, boxes from data loader target
        target = False, boxes from network generate
        """
        aux = COCOAux()
        dis_img = img.numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        dis_img = (dis_img * std + mean) * 255

        dis_img = dis_img.astype(np.uint8).copy()
        h, w = dis_img.shape[0:2]
        boxes = boxes.numpy()
        cls = cls.numpy().astype(np.uint8)
        for box, c in zip(boxes, cls):
            if c == 0: break
            name = aux.int2name(c)
            assert c == aux.id2int(aux.name2id(aux.int2name(c)))
            if target:
                box = DA.cxywh2xywh(box)
                cv2.rectangle(dis_img, (int(box[0] * w), int(box[1] * h)),
                              (int((box[0] + box[2]) * w), int((box[1] + box[3]) * h)), Colors[c], 1)
            else:
                cv2.rectangle(dis_img, (int(box[0] * w), int(box[1] * h)),
                              (int(box[2] * w), int(box[3] * h)), Colors[c], 1)
            cv2.putText(dis_img, name, (int(box[0] * w), int(box[1] * h)), cv2.FONT_HERSHEY_SIMPLEX,
                        1, Colors[c], 1, cv2.LINE_AA)
        return dis_img

    def inference_test(self):
        anchors = np.array([[373, 326], [156, 198], [116, 90],
                            [59, 119], [62, 45], [30, 61],
                            [33, 23], [16, 30], [10, 13]])
        n_classes = 81
        feature_size = [52, 26, 13]
        obj_threshold = 0.5
        score_threshold = -1
        nms_threshold = 0.45
        image_size = 416
        self.inference = YoloInference(anchors=anchors,
                                       n_classes=n_classes,
                                       feature_size=feature_size,
                                       image_size=image_size,
                                       obj_threshold=obj_threshold,
                                       score_threshold=score_threshold,
                                       nms_threshold=nms_threshold)
        plt.figure()
        for i, samples in enumerate(self.loader):
            out = {}
            for k in self.level_keys:
                gt = samples[k]
                o_s = gt.shape[1] - 6  # (no obj,obj)
                _, _, out[k] = gt.split([3, 3, (o_s)], dim=1)
            ind = 0
            gt_image = samples['image'][ind]
            gt_boxes = samples['bboxes'][ind]
            gt_cls = samples['categories'][ind]
            print('index: ', samples['index'][ind])
            # print('gt num: ',gt_cls[gt_cls>0].shape)
            results = self.inference.forward(out)
            result = results[ind]
            boxes, cls, scores = result.split([4, 1, 1], dim=1)
            cls.squeeze_(dim=1)

            dis_gt_img = self.img_writer(gt_image, gt_boxes, gt_cls)
            dis_img = self.img_writer(gt_image, boxes, cls, target=False)

            plt.subplot(1,2,1)
            plt.imshow(dis_gt_img)
            plt.subplot(1,2,2)
            plt.imshow(dis_img)
            plt.pause(1)
            # gt_mask = gt_cls>0
            # gt_boxes = gt_boxes[gt_mask]
            # gt_cls = gt_cls[gt_mask]
            # print(gt_boxes.shape, boxes.shape)
            # assert gt_boxes.shape == boxes.shape
            # assert gt_cls.shape == cls.shape






def main(run='test'):
    if run in ['vis']:
        vis = YoloDataVis()
        vis.ann_visualization()
        vis.target_visualization()
    elif run in ['test']:
        test = YOLOTest()
        # test.loss_test()
        test.inference_test()
    else:
        print('run must be ["test", "vis"]!!')

if __name__ == "__main__":
    import fire

    fire.Fire(main)