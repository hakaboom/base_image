# -*- coding: utf-8 -*-
from .utils import AutoIncrement

SHOW_INDEX = AutoIncrement()


class Place(object):
    Ndarray = 0
    Mat = 1
    GpuMat = 2
    UMat = 3
