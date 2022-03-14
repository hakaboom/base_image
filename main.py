"""
python setup.py sdist
twine upload dist/*
"""
import cv2
import time
from baseImage.base_image import Image, Rect
from baseImage.utils.ssim.paddle import ssim as p_ssim
from baseImage.utils.ssim import ssim
from baseImage.constant import Place

import numpy as np

place_list = [(Place.Ndarray, np.ndarray), (Place.Mat, cv2.Mat), (Place.UMat, cv2.UMat), (Place.GpuMat, cv2.cuda_GpuMat)]

dtype_list = [np.uint8, np.int8, np.uint16, np.int16, np.int32, np.float32, np.float64]

im1 = Image(data='tests/image/1.png', place=Place.Ndarray).resize(640, 692)
im2 = Image(data='tests/image/2.png', place=Place.Ndarray).resize(640, 692)

print(ssim(im1, im2))
# for i in range(100):
#     start = time.time()
#     print(ssim(im1, im2))