import os
import time
import cv2
import numpy as np

from .exceptions import ReadImageError


def check_file(fileName: str):
    """check file in path"""
    return os.path.isfile('{}'.format(fileName))


def check_image_valid(image):
    """检查图像是否有效"""
    if image is not None and image.any():
        return True
    else:
        return False


def read_image(filename: str, flags: int = cv2.IMREAD_COLOR):
    """cv2.imread的加强版"""
    if check_file(filename) is False:
        raise ReadImageError("File not found in path:'{}''".format(filename))

    img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags)

    if check_image_valid(img):
        return img
    else:
        raise ReadImageError('cv2 decode Error, path:{}, flags={}', filename, flags)


def bytes_2_img(byte) -> np.ndarray:
    """bytes转换成cv2可读取格式"""
    img = cv2.imdecode(np.array(bytearray(byte)), 1)
    if img is None:
        raise ValueError('decode bytes to image error, param=\n\'{}\''.format(byte))

    return img


class auto_increment(object):
    def __init__(self):
        self._val = 0

    def __call__(self):
        self._val += 1
        return self._val