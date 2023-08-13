# -*- coding: utf-8 -*-
import cv2
import numpy as np

from baseImage import Image
from baseImage.utils.ssim import SSIM
from typing import Union, Optional, Callable


class ImageDiff(object):
    def __init__(self, win_size: int = 7, data_range: int = 255, sigma: Union[int, float] = 1.5,
                 use_sample_covariance=True, resize=(500, 500), threshold: float = 0.90):
        """
        基于ssim的图片差异获取
        """
        self.ssim = SSIM(resize=resize, win_size=win_size, sigma=sigma, data_range=data_range,
                         use_sample_covariance=use_sample_covariance)
        self.threshold = threshold

    @classmethod
    def _image_check(cls, im1, im2):
        if im1.place != im2.place:
            im2 = Image(im2, place=im1.place, dtype=np.float32)

        if im1.dtype != np.float64:
            im1 = Image(im1, place=im1.place, dtype=np.float64)

        if im2.dtype != np.float64:
            im2 = Image(im2, place=im2.place, dtype=np.float64)

        return im1, im2

    def morphologyFun(self, threshData: Image):
        # 闭运算
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        if threshData.place == cv2.cuda.GpuMat:
            threshData = Image(threshData.data.download())
        erosion = cv2.morphologyEx(threshData.data, cv2.MORPH_ERODE, kernel)
        # erosion = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel2, iterations=2)
        return erosion

    def diff(self, im1: Image, im2: Image, debug: bool = False, threshold: Optional[float] = None):
        return self._diff(im1, im2, debug=debug, threshold=(threshold or self.threshold))

    def _diff(self, im1: Image, im2: Image, threshold, debug: bool = False, morphologyFun: Optional[Callable] = None):
        """
        ssim对比,并找到差异区域

        Args:
            im1: 对比图片1
            im2: 对比图片2
            threshold: 允许的阈值,用于二值化时过滤部分像素,计算公式:thresh=int(255 * (1-threshold))

        Returns:
            tuple|list: 差异区域的轮廓
        """
        mssim, score = self.ssim.ssim(im1, im2, full=True)
        # 灰度
        if score.channels == 3:
            gary = score.cvtColor(cv2.COLOR_BGR2GRAY)
        else:
            gary = score

        # 反色
        gary = gary.bitwise_not()

        # 二值化
        thresh = gary.threshold(code=cv2.THRESH_BINARY, thresh=int(255 * (1 - threshold)), maxval=255)

        # 腐蚀膨胀等运算
        erosion = morphologyFun and morphologyFun(thresh) or self.morphologyFun(thresh)

        # 寻找轮廓
        cnts, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = list(cnts)
        if im1.size != self.ssim.resize:
            scale = np.array([im1.size[1]/self.ssim.resize[1], im1.size[0]/self.ssim.resize[0]])
            for index, cnt in enumerate(cnts):
                cnts[index] = (cnt * scale).astype(np.int32)

        if debug:
            print(f"相似度:{mssim}")
            score.imshow('score')
            thresh.imshow('thresh')
            Image(erosion).imshow('erosion')

        result = []
        for cnt in cnts:
            M = cv2.moments(cnt)
            if M['m00'] == 0.0:
                continue
            else:
                result.append(cnt)

        # for cnt in cnts:
        #     M = cv2.moments(cnt)
        #     print(M)
        return result

"""
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

# show the output images
cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.waitKey(0)"""

if __name__ == '__main__':
    img1 = Image('tests/image/1.png', place=Place.Ndarray)  # .crop(Rect(871,254,647,516))
    img2 = Image('tests/image/2.png', place=Place.Ndarray)  # .crop(Rect(871,254,647,516))

    diff = ImageDiff(resize=img1.size)

    cnts = diff.diff(img1, img2)
    imageA = img1.data.copy()
    imageB = img2.data.copy()
    print(len(cnts))
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Original", imageA)
    cv2.imshow("Modified", imageB)
    cv2.waitKey(0)
