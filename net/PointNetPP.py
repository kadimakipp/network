#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: PointNetPP.py
@time: 2019/12/2 下午4:27
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

class Operator(object):
    @classmethod
    def farthest_points_sample(cls,pts, n_point):
        """
        C - channel, N- Point total number
        get farthest points sample
        :param pts: points (B,N,C)
        :param n_point: sample target number
        :return: idx (B, n_point)
        """
        B, N, C = pts.shape
        device = pts.device
        centroids = torch.zeros((B, n_point), dtype=torch.long,device=device)
        farthest = torch.randint(0, N,(B,),dtype=torch.long)
        distances = torch.ones((B,N))*1e10
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(n_point):
            centroids[:, i] = farthest
            centroid = pts[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((pts - centroid)**2, -1)
            mask = dist<distances
            distances[mask] = dist[mask]
            farthest = torch.max(distances, dim=-1)[1]
        return centroids

    @classmethod
    def indexes2points(cls,pts, indexes):
        """
        :param pts: (B, N, C)
        :param indexes: (B, M)
        :return:
        """
        indexesL = torch.split(indexes, 1, dim=0)
        points = []
        for i, ind in enumerate(indexesL):
            ind.squeeze_(dim=0)
            points.append(pts[i, ind, :].unsqueeze(dim=0))
        return torch.cat(points, dim=0)

    @classmethod
    def euclid_distance(cls, src, dst):
        """
        Calculate Euclid distance between each two points.

        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
             = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B,N,C = src.shape
        _,M,_ = dst.shape
        dist = -2*torch.matmul(src, dst.permute(0,2,1))
        dist += torch.sum(src**2, dim=-1).view(B,N,1)
        dist += torch.sum(dst**2, dim=-1).view(B,1,M)

        return dist

    @classmethod
    def query_ball_point(cls, radius, nsample, xyz, new_xyz):
        """
        Input:
           radius: local region radius
           nsample: max sample number in local region
           xyz: all points, [B, N, C]
           new_xyz: query points, [B, S, C]
        Return:
           group_idx: grouped points index, [B, S, nsample]
        """
        device = xyz.device
        B, N, C = xyz.shape
        _, S, _ = new_xyz.shape
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        sqrdists = cls.euclid_distance(new_xyz, xyz)
        group_idx[sqrdists > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
        mask = group_idx == N
        group_idx[mask] = group_first[mask]
        return group_idx

class ConvBlock(nn.Sequential):
    def __init__(self, inc, ouc):
        super(ConvBlock, self).__init__(
            nn.Conv1d(inc,ouc,kernel_size=1),
            nn.BatchNorm1d(ouc),
            nn.ReLU(inplace=True)
        )

class PointNetLayer(nn.Module):
    def __init__(self, inc, ouc):
        assert isinstance(ouc, list)
        super(PointNetLayer, self).__init__()
        last_inc = inc
        conv_list = []
        for ou in ouc:
            conv_list.append(ConvBlock(last_inc, ou))
            last_inc = ou
        self.net = nn.Sequential(*conv_list)

    def forward(self, x):
        x = self.net(x)
        x = torch.max(x,dim=-1)[0].unsqueeze(dim=-1)
        return x

def main():

    pts = torch.randn((2,15,3))
    print(pts.shape)
    indexes = Operator.farthest_points_sample(pts,7)
    print(indexes.shape)
    new_pts = Operator.indexes2points(pts, indexes)
    print(new_pts.shape)
    # pts = torch.randn(2,3,1024)
    # net = PointNetLayer(3, [64,128,256])
    # out = net(pts)
    # print(out.shape)

if __name__ == "__main__":
    import fire

    fire.Fire(main)