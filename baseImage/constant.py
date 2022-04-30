# -*- coding: utf-8 -*-
import cv2
from .utils.api import AutoIncrement

SHOW_INDEX = AutoIncrement()


class Place(object):
    Ndarray = 0
    Mat = 1
    GpuMat = 2
    UMat = 3


if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    CUDA_Flag = True
    Default_Stream = cv2.cuda.Stream()
else:
    CUDA_Flag = False
    Default_Stream = None


operations = {
    'mat': {
        'multiply': cv2.multiply,
        'subtract': cv2.subtract,
        'add': cv2.add,
        'pow': cv2.pow,
        'divide': cv2.divide,
        'merge': cv2.merge,
    },
}

if CUDA_Flag:
    operations['cuda'] = {
        'multiply': cv2.cuda.multiply,
        'subtract': cv2.cuda.subtract,
        'add': cv2.cuda.add,
        'pow': cv2.cuda.pow,
        'divide': cv2.cuda.divide,
        'merge': cv2.cuda.merge,
    }
