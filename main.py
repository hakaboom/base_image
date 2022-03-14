"""
python setup.py sdist
twine upload dist/*
"""
import cv2
import time
import paddle
from baseImage.base_image import Image, Rect
from baseImage.utils import ssim, np_ssim, umat_ssim
from baseImage.constant import Place

import numpy as np
print(cv2.__version__)
place_list = [(Place.Ndarray, np.ndarray), (Place.Mat, cv2.Mat), (Place.UMat, cv2.UMat), (Place.GpuMat, cv2.cuda_GpuMat)]

dtype_list = [np.uint8, np.int8, np.uint16, np.int16, np.int32, np.float32, np.float64]

im1 = Image(data='test/1.png', dtype=np.float32, place=Place.Ndarray).resize(1280, 692)

