import os

import cv2
import numpy as np


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
        raise IOError('File not found, path:{}'.format(filename))
    img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags)
    if check_image_valid(img):
        return img
    else:
        raise Exception('cv2 decode Error, path:{}, flage={}', filename, flags)


def bytes_2_img(byte) -> np.ndarray:
    """bytes转换成cv2可读取格式"""
    img = cv2.imdecode(np.array(bytearray(byte)), 1)
    if img is None:
        raise ValueError('decode bytes to image error \n\'{}\''.format(byte))
    return img


def bgr_2_gray(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    gray_img = 0.2126 * r + 0.7152 * g + 0.0722 * b
    gray_img = gray_img.astype(np.uint8)

    return gray_img


class auto_increment(object):
    def __init__(self):
        self._val = 0

    def __call__(self):
        self._val += 1
        return self._val