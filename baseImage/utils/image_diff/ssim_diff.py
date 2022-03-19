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

    @classmethod
    def _image_check(cls, im1, im2):
        if im1.place != im2.place:
            im2 = Image(im2, place=im1.place, dtype=np.float32)

        if im1.dtype != np.float32:
            im1 = Image(im1, place=im1.place, dtype=np.float32)

        if im2.dtype != np.float32:
            im2 = Image(im2, place=im2.place, dtype=np.float32)

        return im1, im2

    def diff(self, im1: Image, im2: Image):
        im1, im2 = self._image_check(im1, im2)
        return self._diff(im1, im2)

    def _diff(self, im1: Image, im2: Image):
        mssim, score = self.ssim.ssim(im1, im2, full=True)
        # 灰度
        if score.channels == 3:
            gary = score.cvtColor(cv2.COLOR_BGR2GRAY)
        else:
            gary = score
        Image(gary).imshow()

        # 二值化
        thresh = gary.bitwise_not().threshold(code=cv2.THRESH_BINARY)
        Image(thresh).imshow('thresh')

        # 闭运算
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        erosion = cv2.morphologyEx(thresh.data, cv2.MORPH_CLOSE, kernel)

        # 寻找轮廓
        cnts, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if im1.size != self.ssim.resize:
            new_cnts = []
            scale = np.array([im1.size[1]/self.ssim.resize[1], im1.size[0]/self.ssim.resize[0]])
            for cnt in cnts:
                cnt = cnt * scale
                cnt = cnt.astype(np.int32)
                new_cnts.append(cnt)
            return new_cnts
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