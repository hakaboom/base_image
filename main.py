"""
python setup.py sdist
twine upload dist/*
"""
from baseImage import Image, Rect, Size
from baseImage.test_base_image import Image
import paddle
import cv2
import time
import numpy as np
from typing import Union

from functools import singledispatchmethod
#
for i in range(3):
    a = Image('./test/0.png', place=2)
    print(a.crop(Rect(1, 1, 200, 200)).size)
# a.resize(1)
# a.transform_gpu()
# b = a.download()
#
# a.crop_image(Rect(500, 500, 500, 500)).imshow()
# c = cv2.cuda_GpuMat(b).adjustROI(500, 500, 500, 500)
#
# c = Image(c)
# c.imshow()
# cv2.waitKey(0)

# data = a.imread()
#
# b =
#
# detector = cv2.ORB_create(nfeatures=50000)
