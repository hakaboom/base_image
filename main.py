"""
python setup.py sdist
twine upload dist/*
"""
import cv2
import time
import os
from baseImage.base_image import Image, Rect
from baseImage.utils.ssim import SSIM
from baseImage.constant import Place
from baseImage.utils.image_diff.ssim_diff import ImageDiff

import numpy as np


img1 = Image('tests/image/test1.png', place=Place.GpuMat)
img2 = Image('tests/image/test2.png', place=Place.GpuMat)

diff = ImageDiff()

cnts = diff.diff(img1, img2)
imageA = img1.data.download()
imageB = img2.data.download()

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

print(len(cnts))
cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.waitKey(0)
