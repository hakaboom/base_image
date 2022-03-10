"""
python setup.py sdist
twine upload dist/*
"""
import os

from baseImage import Image, Rect, Size
from baseImage.test_base_image import Image
from baseImage.utils import read_image, cvType_to_npType, npType_to_cvType
import paddle
import cv2
import numpy as np
import threading
import queue
import time
print(cv2.__version__)
a = Image('test/1.png', place=2, dtype=np.float32).data
data_type = cvType_to_npType(a.type(), channel=a.channels())


# 目标
dtype = np.uint8
if data_type != dtype:
    print(f'当前类型为{data_type}, 目标类型{dtype}')
    b = cv2.cuda.GpuMat(a.size(), npType_to_cvType(dtype, a.channels()))
    a.convertTo(npType_to_cvType(dtype, a.channels()), b)
    print(b.download().dtype, a.download().dtype)
# for i in range(4):
#     a = Image('test/1.png', place=i)
#     for _i in range(4):
#         b = Image(a.data, place=i)