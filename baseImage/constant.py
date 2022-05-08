# -*- coding: utf-8 -*-
import cv2
import numpy as np

from .utils.api import AutoIncrement

SHOW_INDEX = AutoIncrement()


class Place(object):
    Ndarray = np.ndarray
    GpuMat = cv2.cuda.GpuMat
    UMat = cv2.UMat


class Setting(object):
    CUDA_Flag = False
    Default_Stream = None
    Default_Pool = None
    Default_Place = Place.Ndarray


if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    Setting.CUDA_Flag = True
    Setting.Default_Stream = cv2.cuda.Stream()
    Setting.Default_Pool = None


operations = {
    'mat': {
        'multiply': cv2.multiply,
        'subtract': cv2.subtract,
        'add': cv2.add,
        'pow': cv2.pow,
        'divide': cv2.divide,
        'merge': cv2.merge,
    },
    'cuda': {
        'multiply': None,
        'subtract': None,
        'add': None,
        'pow': None,
        'divide': None,
        'merge': None,
    }
}

if Setting.CUDA_Flag:
    operations['cuda'] = {
        'multiply': cv2.cuda.multiply,
        'subtract': cv2.cuda.subtract,
        'add': cv2.cuda.add,
        'pow': cv2.cuda.pow,
        'divide': cv2.cuda.divide,
        'merge': cv2.cuda.merge,
    }
