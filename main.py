"""
python setup.py sdist
twine upload dist/*
"""
import time
import os
import numpy as np

from baseImage import Image, Rect
from baseImage.utils.ssim import SSIM

import cv2

path = r'C:\Users\Administrator.hzq\Desktop\test\RPReplay_Final1648280745.MP4'
save_path = r'C:\Users\Administrator.hzq\Desktop\test'

video = cv2.VideoCapture(path)
index = 0
frame_index = 1
if video.isOpened():
    rval, frame = video.read()
else:
    rval = False

while rval:
    rval, frame = video.read()

    if index % 10 == 0:
        Image(frame).imwrite(os.path.join(save_path, f'{int(frame_index)}.png'))
        frame_index += 1
    index += 1