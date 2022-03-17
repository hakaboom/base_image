# -*- coding: utf-8 -*-
import cv2
import numpy as np

from baseImage import Image
from baseImage.utils.ssim import SSIM


class ImageDiff(object):
    def __init__(self, resize=(500, 500)):
        """
        基于ssim的图片差异获取
        """
        self.ssim = SSIM(resize=resize)

    def diff(self, im1: Image, im2: Image):
        mssim, score = self.ssim.ssim(im1, im2, full=True)
        # 灰度
        gary = score.cvtColor(cv2.COLOR_BGR2GRAY)
        gary.dtype_convert(dtype=np.uint8)
        # 二值化
        thresh = gary.threshold(code=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU).data.download()

        # 闭运算
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        erosion = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 寻找轮廓
        cnts, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if im1.size != self.ssim.resize:
            scale = np.array([im1.size[1]/self.ssim.resize[1], im1.size[0]/self.ssim.resize[0]])
            cnts = cnts * scale
            cnts = cnts.astype(np.int32)
        return cnts

"""
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

# show the output images
cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.waitKey(0)"""