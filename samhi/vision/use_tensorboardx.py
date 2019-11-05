#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: use_tensorboardx.py
@time: 2019/11/5 上午10:22
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torchvision
import torchvision.utils as vutils
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter

def tensorboardX_demo():
    resnet18 = models.resnet18(False)
    writer = SummaryWriter()
    sample_rate = 44100
    freqs = [262,294,330, 349,392,440, 440,440,440, 440,440]
    for n_iter in range(100):
        dummy_s1 = torch.rand(1)
        dummy_s2 = torch.rand(1)
        #data grouping by 'slash'
        writer.add_scalar('data/scalar1', dummy_s1[0], n_iter)
        writer.add_scalar('data/scalar2', dummy_s2[0], n_iter)
        writer.add_scalars('data/scalar_group', {'xsinx': n_iter*np.sin(n_iter),
                                                 'xcosx': n_iter*np.cos(n_iter),
                                                 'arctanx': n_iter*np.arctan(n_iter)}, n_iter)

        dummy_img = torch.rand(32, 3, 64, 64) #output from network
        if n_iter % 10 == 0:
            x = vutils.make_grid(dummy_img, normalize=True, scale_each=True)
            writer.add_image('Image', x, n_iter)

            dummy_audio = torch.zeros(sample_rate*2)
            for i in range(x.size(0)):
                #amplitude of sound should in [-1, 1]
                dummy_audio[i] = np.cos(freqs[n_iter//10]*np.pi*float(i)/float(sample_rate))
            writer.add_audio('myAudio', dummy_audio, n_iter, sample_rate=sample_rate)
            writer.add_text('Text', 'Text logged at step: '+ str(n_iter), n_iter)

            for name, param in resnet18.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)

            rand1000 = np.arange(1000)/1000
            writer.add_histogram('distribution centers', rand1000 + n_iter/100, n_iter)


                # needs tensorboard 0.4RC or later
            writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(100), n_iter)

    dataset = datasets.MNIST('/media/kipp/work/Datas', train=False, download=False)
    images = dataset.test_data[:100].float()
    label = dataset.test_labels[:100]

    features = images.view(100, 784)
    writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

def main():
    tensorboardX_demo()


if __name__ == "__main__":
    import fire

    fire.Fire(main)