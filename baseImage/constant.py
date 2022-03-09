# -*- coding: utf-8 -*-
from .utils import auto_increment

SHOW_INDEX = auto_increment()


class Place(object):
    Ndarray = 0
    Mat = 1
    GpuMat = 2
