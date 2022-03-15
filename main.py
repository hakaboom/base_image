"""
python setup.py sdist
twine upload dist/*
"""
import cv2
import time
from baseImage.base_image import Image, Rect
from baseImage.utils.ssim.paddle import ssim as p_ssim
from baseImage.utils.ssim import SSIM
from baseImage.constant import Place

import numpy as np

place_list = [(Place.Ndarray, np.ndarray), (Place.Mat, cv2.Mat), (Place.UMat, cv2.UMat), (Place.GpuMat, cv2.cuda_GpuMat)]

dtype_list = [np.uint8, np.int8, np.uint16, np.int16, np.int32, np.float32, np.float64]


a = SSIM()
im1 = Image('tests/image/1.png', dtype=np.float32, place=Place.Ndarray)
im2 = Image('tests/image/2.png', dtype=np.float32, place=Place.Ndarray)

# 0.6839703575015336
for i in range(1000):
    start = time.time()
    a.ssim(im1, im2)
    print(time.time() - start)

# im2 = Image('tests/image/2.png', dtype=np.float32)
