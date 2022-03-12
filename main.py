"""
python setup.py sdist
twine upload dist/*
"""
import cv2
from baseImage.base_image import Image, Rect
from baseImage.constant import Place

place_list = [Place.Ndarray, Place.Mat, Place.UMat, Place.GpuMat]


img = Image(data='tests/image/0.png', place=Place.GpuMat)
