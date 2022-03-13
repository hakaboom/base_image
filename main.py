"""
python setup.py sdist
twine upload dist/*
"""
import cv2

from baseImage.base_image import Image, Rect
from baseImage.constant import Place
from baseImage.utils import cvType_to_npType, npType_to_cvType
import numpy as np
print(cv2.__version__)
place_list = [(Place.Ndarray, np.ndarray), (Place.Mat, cv2.Mat), (Place.UMat, cv2.UMat), (Place.GpuMat, cv2.cuda_GpuMat)]

dtype_list = [np.uint8, np.int8, np.uint16, np.int16, np.int32, np.float32, np.float64]

img1 = Image(data='tests/image/1.png', dtype=np.float32, place=Place.Ndarray).split()
img2 = Image(data='tests/image/2.png', dtype=np.float32, place=Place.Ndarray).split()

img1 = Image(data=img1[0], dtype=np.float32, place=Place.Ndarray)
img2 = Image(data=img2[0], dtype=np.float32, place=Place.Ndarray)
img1.imshow()
img2.imshow()
cv2.waitKey(0)
K1 = 0.01
K2 = 0.03
win_size = 11
sigma = 1.5
truncate = 3.5
data_range = 255

NP = win_size ** 3

cov_norm = NP / (NP - 1)

k_size = int(truncate * sigma)

ux = img1.gaussianBlur((k_size, k_size), sigma).data
uy = img2.gaussianBlur((k_size, k_size), sigma).data

uxx = Image(data=(ux * ux), clone=False).gaussianBlur((k_size, k_size), sigma).data
uyy = Image(data=(uy * uy), clone=False).gaussianBlur((k_size, k_size), sigma).data

uxy = Image(data=(ux * uy), clone=False).gaussianBlur((k_size, k_size), sigma).data


vx = cov_norm * (uxx - ux * ux)
vy = cov_norm * (uyy - uy * uy)
vxy = cov_norm * (uxy - ux * uy)

R = data_range
C1 = (K1 * R) ** 2
C2 = (K2 * R) ** 2

A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
D = B1 * B2
S = (A1 * A2) / D

pad = (win_size - 1) // 2
h, w = S.shape[:2]

data = Image(data=S, clone=False).crop(Rect(5, 5, w - 10, h - 10))
mssim = data.data.mean(dtype=np.float64)
print(mssim)