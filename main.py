"""
python setup.py sdist
twine upload dist/*
"""
import cv2

from baseImage.base_image import Image, Rect
from baseImage.constant import Place
from baseImage.utils import cvType_to_npType
import numpy as np
print(cv2.__version__)
place_list = [(Place.Ndarray, np.ndarray), (Place.Mat, cv2.Mat), (Place.UMat, cv2.UMat), (Place.GpuMat, cv2.cuda_GpuMat)]

dtype_list = [np.uint8, np.int8, np.uint16, np.int16, np.int32, np.float32, np.float64]


img = Image(data='tests/image/0.png', place=Place.GpuMat)
img.dtype_convert(dtype=np.float32)