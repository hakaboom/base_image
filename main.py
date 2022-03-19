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
import matplotlib.pyplot as plt

import numpy as np


img1 = Image('tests/image/test1.png', place=Place.Ndarray)  # .crop(Rect(871,254,647,516))
img2 = Image('tests/image/test2.png', place=Place.Ndarray)  # .crop(Rect(871,254,647,516))


diff = ImageDiff(img1.size)

cnts = diff.diff(img1, img2)
imageA = img1.data.copy()
imageB = img2.data.copy()

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
# img1.imshow()
# img2.imshow()
cv2.waitKey(0)
