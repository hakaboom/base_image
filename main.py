"""
python setup.py sdist
twine upload dist/*
"""
import os

from baseImage import Image, Rect, Size
from baseImage.test_base_image import Image
from baseImage.utils import read_image, cvType_to_npType, npType_to_cvType
import paddle
import cv2
import numpy as np
import threading
import queue
import time
from loguru import logger


a = Image('test/0.png', place=3)
