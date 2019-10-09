#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: kipp
@contact: kaidma.kipp@gmail.com
@site: 
@software: PyCharm
@file: klog.py
@time: 2019/10/9 上午10:55
# Shallow men believe in luck.
Strong men believe in cause and effect.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from pathlib import Path
import time

class KLOG(object):
    def __init__(self, logger):
        """root - file save dir
        """
        root = os.path.dirname(__file__)
        fire = os.path.join(root, "fire")
        self.root = Path(os.path.join(fire, 'logs'))
        self.root.mkdir(exist_ok=True)
        self.logger = logging.getLogger(logger)
        formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s- %(message)s')
        self.logger.setLevel(logging.DEBUG)
        name = time.strftime('%b-%d-%H-%M-%S', time.localtime())
        logfile = self.root.joinpath(name+'.log')
        file_handler = logging.FileHandler(str(logfile))
        file_handler.setFormatter(formatter)
        sh_handler = logging.StreamHandler()  # redefine stream
        formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s- %(message)s')
        sh_handler.setFormatter(formatter)  # control output
        self.logger.addHandler(sh_handler)
        self.logger.addHandler(file_handler)

    def warning(self,msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def WARNING(self, msg, *args, **kwargs):
        self.warning(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def INFO(self, msg, *args, **kwargs):
        self.info(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def ERROR(self, msg, *args, **kwargs):
        self.error(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def DENUG(self, msg, *args, **kwargs):
        self.debug(msg, *args, **kwargs)

def main():
    klog = KLOG('kipp')
    klog.warning('testing')
    klog.error('test %s ', __file__)
    klog.info("test %d",1000)
    klog.debug('test %d',10)


if __name__ == "__main__":
    import fire

    fire.Fire(main)