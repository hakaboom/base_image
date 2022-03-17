"""
python setup.py sdist
twine upload dist/*
"""
import cv2
import time
import os
from baseImage.base_image import Image, Rect
from baseImage.utils.ssim.paddle import ssim as p_ssim
from baseImage.utils.ssim import SSIM
from baseImage.constant import Place

import numpy as np

path = 'tests/image/test.mp4'
save_path = 'tests/image/test/'

vc = cv2.VideoCapture(path)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False
c = 1
while rval:
    rval, frame = vc.read()
    cv2.imwrite(os.path.join(save_path, f'{str(c)}.png'), Image(frame).data)
    c += 1
    cv2.waitKey(1)
vc.release()