"""
python setup.py sdist
twine upload dist/*
"""
import os

from baseImage import Image, Rect, Size
from baseImage.test_base_image import Image
from baseImage.utils import read_image
import paddle
import cv2
import numpy as np
import threading
import queue
import time

a = Image('test/1.png', place=1)
# b = Image(a.rectangle(rect=Rect(1, 1, 200, 200)))
# a.imshow()
# b.imshow()
# cv2.waitKey(0)

stream = cv2.cuda.Stream()
start = time.time()
b = []
for i in range(1000):
    b.append(a.data.copy())
print(time.time() - start)

for i in range(1000):
    cuMat = cv2.cuda.GpuMat()
    cuMat.upload(b[i], stream)
print(stream.queryIfComplete())
# stream.waitForCompletion()
print(time.time() - start)
stream.waitForCompletion()
print(time.time() - start)