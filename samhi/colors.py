#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: colors.py
@time: 2019/11/14 上午9:18
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
RGB_COLORS=[
(255,182,193),#LightPink 浅粉红
    (255,192,203),#Pink 粉红
    (220,20,60),#Crimson 猩红
    (255,240,245),#LavenderBlush 脸红的淡紫色
    (219,112,147),#PaleVioletRed 苍白的紫罗兰红色
    (255,105,180),#HotPink 热情的粉红
    (255,20,147),#DeepPink 深粉色
    (199,21,133),#MediumVioletRed 适中的紫罗兰红色
    (218,112,214),#Orchid 兰花的紫色
    (216,191,216),#Thistle 蓟
    (221,160,221),#plum 李子
    (238,130,238),#Violet 紫罗兰
    (255,0,255),#Magenta 洋红
    (255,0,255),#Fuchsia 灯笼海棠(紫红色)
    (139,0,139),#DarkMagenta 深洋红色
    (128,0,128),#Purple 紫色
    (186,85,211),#MediumOrchid 适中的兰花紫
    (148,0,211),#DarkVoilet 深紫罗兰色
    (153,50,204),#DarkOrchid 深兰花紫
    (75,0,130),#Indigo 靛青
    (138,43,226),#BlueViolet 深紫罗兰的蓝色
    (147,112,219),#MediumPurple 适中的紫色
    (123,104,238),#MediumSlateBlue 适中的板岩暗蓝灰色
    (106,90,205),#SlateBlue 板岩暗蓝灰色
    (72,61,139),#DarkSlateBlue 深岩暗蓝灰色
    (230,230,250),#Lavender 熏衣草花的淡紫色
    (248,248,255),#GhostWhite 幽灵的白色
    (0,0,255),#Blue 纯蓝
    (0,0,205),#MediumBlue 适中的蓝色
    (25,25,112),#MidnightBlue 午夜的蓝色
    (0,0,139),#DarkBlue 深蓝色
    (0,0,128),#Navy 海军蓝
    (65,105,225),#RoyalBlue 皇家蓝
    (100,149,237),#CornflowerBlue 矢车菊的蓝色
    (176,196,222),#LightSteelBlue 淡钢蓝
    (119,136,153),#LightSlateGray 浅石板灰
    (112,128,144),#SlateGray 石板灰
    (30,144,255),#DoderBlue 道奇蓝
    (240,248,255),#AliceBlue 爱丽丝蓝
    (70,130,180),#SteelBlue 钢蓝
    (135,206,250),#LightSkyBlue 淡蓝色
    (135,206,235),#SkyBlue 天蓝色
    (0,191,255),#DeepSkyBlue 深天蓝
    (173,216,230),#LightBLue 淡蓝
    (176,224,230),#PowDerBlue 火药蓝
    (95,158,160),#CadetBlue 军校蓝
    (240,255,255),#Azure 蔚蓝色
    (225,255,255),#LightCyan 淡青色
    (175,238,238),#PaleTurquoise 苍白的绿宝石
    (0,255,255),#Cyan 青色
    (0,255,255),#Aqua 水绿色
    (0,206,209),#DarkTurquoise 深绿宝石
    (47,79,79),#DarkSlateGray 深石板灰
    (0,139,139),#DarkCyan 深青色
    (0,128,128),#Teal 水鸭色
    (72,209,204),#MediumTurquoise 适中的绿宝石
    (32,178,170),#LightSeaGreen 浅海洋绿
    (64,224,208),#Turquoise 绿宝石
    (127,255,170),#Auqamarin 绿玉\碧绿色
    (0,250,154),#MediumAquamarine 适中的碧绿色
    (0,255,127),#MediumSpringGreen 适中的春天的绿色
    (245,255,250),#MintCream 薄荷奶油
    (60,179,113),#SpringGreen 春天的绿色
    (46,139,87),#SeaGreen 海洋绿
    (240,255,240),#Honeydew 蜂蜜
    (144,238,144),#LightGreen 淡绿色
    (152,251,152),#PaleGreen 苍白的绿色
    (143,188,143),#DarkSeaGreen 深海洋绿
    (50,205,50),#LimeGreen 酸橙绿
    (0,255,0),#Lime 酸橙色
    (34,139,34),#ForestGreen 森林绿
    (0,128,0),#Green 纯绿
    (0,100,0),#DarkGreen 深绿色
    (127,255,0),#Chartreuse 查特酒绿
    (124,252,0),#LawnGreen 草坪绿
    (173,255,47),#GreenYellow 绿黄色
    (85,107,47),#OliveDrab 橄榄土褐色
    (245,245,220),#Beige 米色(浅褐色)
    (250,250,210),#LightGoldenrodYellow 浅秋麒麟黄
    (255,255,240),#Ivory 象牙
    (255,255,224),#LightYellow 浅黄色
    (255,255,0),#Yellow 纯黄
    (128,128,0),#Olive 橄榄
    (189,183,107),#DarkKhaki 深卡其布
    (255,250,205),#LemonChiffon 柠檬薄纱
    (238,232,170),#PaleGodenrod 灰秋麒麟
    (240,230,140),#Khaki 卡其布
    (255,215,0),#Gold 金
    (255,248,220),#Cornislk 玉米色
    (218,165,32),#GoldEnrod 秋麒麟
    (255,250,240),#FloralWhite 花的白色
    (253,245,230),#OldLace 老饰带
    (245,222,179),#Wheat 小麦色
    (255,228,181),#Moccasin 鹿皮鞋
    (255,165,0),#Orange 橙色
    (255,239,213),#PapayaWhip 番木瓜
    (255,235,205),#BlanchedAlmond 漂白的杏仁
    (255,222,173),#NavajoWhite 纳瓦霍白
    (250,235,215),#AntiqueWhite 古代的白色
    (210,180,140),#Tan 晒黑
    (222,184,135),#BrulyWood 结实的树
    (255,228,196),#Bisque (浓汤)乳脂,番茄等
    (255,140,0),#DarkOrange 深橙色
    (250,240,230),#Linen 亚麻布
    (205,133,63),#Peru 秘鲁
    (255,218,185),#PeachPuff 桃色
    (244,164,96),#SandyBrown 沙棕色
    (210,105,30),#Chocolate 巧克力
    (139,69,19),#SaddleBrown 马鞍棕色
    (255,245,238),#SeaShell 海贝壳
    (160,82,45),#Sienna 黄土赭色
    (255,160,122),#LightSalmon 浅鲜肉(鲑鱼)色
    (255,127,80),#Coral 珊瑚
    (255,69,0),#OrangeRed 橙红色
    (233,150,122),#DarkSalmon 深鲜肉(鲑鱼)色
    (255,99,71),#Tomato 番茄
    (255,228,225),#MistyRose 薄雾玫瑰
    (250,128,114),#Salmon 鲜肉(鲑鱼)色
    (255,250,250),#Snow 雪
    (240,128,128),#LightCoral 淡珊瑚色
    (188,143,143),#RosyBrown 玫瑰棕色
    (205,92,92),#IndianRed 印度红
    (255,0,0),#Red 纯红
    (165,42,42),#Brown 棕色
    (178,34,34),#FireBrick 耐火砖
    (139,0,0),#DarkRed 深红色
    (128,0,0),#Maroon 栗色
    (255,255,255),#White 纯白
    (245,245,245),#WhiteSmoke 白烟
    (220,220,220),#Gainsboro 亮灰色
    (211,211,211),#LightGrey 浅灰色
    (192,192,192),#Silver 银白色
    (169,169,169),#DarkGray 深灰色
    (128,128,128),#Gray 灰色
    (105,105,105),#DimGray 暗淡的灰色
    (0,0,0),#Black 纯黑
]

def main():
    for color in RGB_COLORS:
        print(color)


if __name__ == "__main__":
    import fire

    fire.Fire(main)