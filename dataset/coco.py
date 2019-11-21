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
import torchvision.transforms as transforms
from pycocotools.coco import COCO as COCOtool
from samhi.detected import DetectedAux as DA
"""
https://github.com/BobLiu20/YOLOv3_PyTorch/blob/master/common/coco_dataset.py
"""
class COCOAux(object):
    def __init__(self):
        categories = [ {'name': 'person', 'id': 1, 'supercategory': 'person'}, {'name': 'bicycle', 'id': 2, 'supercategory': 'vehicle'},
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
        coco_ids = [cat['id'] for cat in categories]
        coco_names = [cat['name'] for cat in categories]
        ids = [0]+coco_ids
        names = ['__background__'] + coco_names

        self.min_keypoints_per_image = 10

        self.__id2int = dict(zip(ids, range(len(ids))))
        self.__int2id = dict(zip(range(len(ids)), ids))
        self.__name2int = dict(zip(names, range(len(names))))
        self.__int2name = dict(zip(range(len(names)), names))

    def id2int(self, id:int):
        return self.__id2int[id]

    def int2id(self, ind:int):
        return self.__int2id[ind]

    def name2int(self, name):
        return self.__name2int[name]

    def int2name(self, ind:int):
        return self.__int2name[ind]

    def id2name(self, id):
        return self.int2name(self.id2int(id))

    def name2id(self,name):
        return self.int2id(self.name2int(name))

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
            categories.append(self.id2int(o['category_id']))
            bboxes.append(DA.xywh2cxywh(o['bbox']))
        categories = np.array(categories)
        bboxes = np.array(bboxes)
        return categories,bboxes

    def _count_visible_keypoints(self,anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    def _has_only_empty_bbox(self,anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def has_valid_annotation(self,anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if self._has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different critera for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if self._count_visible_keypoints(anno) >= self.min_keypoints_per_image:
            return True
        return False

    @classmethod
    def image_dir(cls, mode='val', years='2014'):
        assert mode in ['train','val', 'test']
        assert years in ['2014', '2017']
        return mode+years

    @classmethod
    def annotation_path(cls, years,prefix):
        assert years in ['2014', '2017']
        assert prefix in ['train','val', 'test', 'minval']

        return "annotations{}/instances_{}{}.json".format(years,prefix,years)

class COCO(Dataset):
    def __init__(self, root, transform, target_transform, train, years):
        super(COCO, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.annFile = os.path.join(self.root, COCOAux.annotation_path(years,train))
        self.aux = COCOAux()
        self.img_dir = os.path.join(self.root, COCOAux.image_dir(train,years))

        self.coco = COCOtool(self.annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

        # filter images without detection annotations
        ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if self.aux.has_valid_annotation(anno):
                ids.append(img_id)
        self.ids = ids

    def __len__(self):
        return len(self.ids)

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
        # img = Image.open(os.path.join(self.img_dir, file_name)).convert('RGB')
        img_path = os.path.join(self.img_dir, file_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception("check image {}".format(img_path))
        o_h, o_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        categories,bboxes = self.aux.parse_coco_ann(target)
        if len(bboxes.shape) == 1:
            #should be in image is have no obj...
            raise Exception("check bboxes shape {} in image id {}, {}".format(bboxes.shape,index ,img_path))
        sample={'image':img,
                'bboxes':bboxes,
                'categories':categories,
                'path':img_path,
                'o_hw':(o_h,o_w),
                'index':index
                }
        sample = self.transform(sample)

        return sample

from dataset import selfT

class CoCo(object):
    def __init__(self, max_object=64):
        # self.root = '/media/kipp/data/DATASET/COCO'
        self.root = '/media/kipp/work/DATASET/COCO'
        self.num_work = 4
        self.shuffle = True
        self.max_object = max_object
        self.anchors = np.array([[373, 326], [156, 198], [116, 90],
                                 [59, 119], [62, 45], [30, 61],
                                 [33, 23], [16, 30], [10, 13]])
        self.feature_size = np.array([52, 26, 13])
        self.n_classes = 81

    def Transform(self, img_size):
        transform = [
            # transforms.RandomCrop(224),
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomAffine(5),
            selfT.KeepAspect(),
            selfT.Resize(img_size),
            selfT.YoloTarget(self.n_classes, self.anchors, img_size, self.feature_size),
            selfT.ToTensor(self.max_object, self.feature_size),
            selfT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        return transforms.Compose(transform)


    def get_loader(self, batch_size, img_size, mode='val',years='2017'):
        transform= self.Transform(img_size)
        return torch.utils.data.DataLoader(
            COCO(self.root, transform=transform,target_transform=transform,train=mode, years=years),
            batch_size=batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_work
        )

import matplotlib.pyplot as plt
from samhi.colors import RGB_COLORS
def img_writer(img, boxes, cls):
    aux = COCOAux()
    dis_img = img.numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    dis_img = (dis_img*std+mean)*255

    dis_img = dis_img.astype(np.uint8).copy()
    h,w= dis_img.shape[0:2]
    boxes = boxes.numpy()
    cls = cls.numpy().astype(np.uint8)
    for box, c in zip(boxes, cls):
        if c == 0:break
        name = aux.int2name(c)
        assert c == aux.id2int(aux.name2id(aux.int2name(c)))
        box = DA.cxywh2xywh(box)
        cv2.rectangle(dis_img, (int(box[0]*w), int(box[1]*h)),
                      (int((box[0]+box[2])*w),int((box[1]+box[3])*h)),RGB_COLORS[c], 1)
        cv2.putText(dis_img, name, (int(box[0] * w), int(box[1] * h)), cv2.FONT_HERSHEY_SIMPLEX,
                    1, RGB_COLORS[c], 2, cv2.LINE_AA)
    return dis_img

def main():
    coco = CoCo()
    loader = coco.get_loader(2,416)
    print(len(loader))

    for i, samples in enumerate(loader):
        images = samples['image']
        boxes = samples['bboxes']
        cls = samples['categories']
        if 'scale' in samples.keys():
            for f_s in [52, 26, 13]:
                key_str = 'scale_{}_'.format(f_s)
                for k in ['no_obj', 'obj', 'target']:
                    print(samples[key_str + k].shape)
            print(images.shape, boxes.shape, cls.shape)
        dis_img = img_writer(images[0], boxes[0], cls[0])
        plt.imshow(dis_img)
        plt.show()
        # break

if __name__ == "__main__":
    import fire

    fire.Fire(main)