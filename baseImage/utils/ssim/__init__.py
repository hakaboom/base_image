from baseImage import Image, Rect
from baseImage.constant import Place

import cv2
import numpy as np


multiply = cv2.multiply
subtract = cv2.subtract
add = cv2.add
pow = cv2.pow
divide = cv2.divide


class SSIM(object):
    def __init__(self, win_size: int = 11, data_range: int = 255, sigma: int = 1.5, use_sample_covariance=True):
        self.win_size = win_size
        self.data_range = data_range
        self.sigma = sigma
        self.dtype = np.float32
        self.K1 = 0.01
        self.K2 = 0.03
        NP = win_size ** 2

        # filter has already normalized by NP
        if use_sample_covariance:
            cov_norm = NP / (NP - 1)  # sample covariance
        else:
            cov_norm = 1.0  # population covariance to match Wang et. al. 2004
        self.cov_norm = cov_norm
        self.gaussian_args = {'size': (win_size, win_size), 'sigma': sigma, 'borderType': cv2.BORDER_REFLECT}

    @classmethod
    def _image_check(cls, im1: Image, im2: Image):
        if not isinstance(im1, Image) and not isinstance(im2, Image):
            raise ValueError('im1 im2必须为Image类型, im1_type:{}, im2_type:{}'.format(type(im1), type(im2)))

        if im1.place != im2.place:
            raise ValueError('图片类型必须一致, im1:{}, im2:{}'.format(im1.place, im2.place))

        if im1.channels != im2.channels:
            raise ValueError('图片通道数量必须一致, im1:{}, im2:{}'.format(im1.place, im2.place))

        if im1.size != im2.size:
            raise ValueError('图片通道数量大小一致, im1:{}, im2:{}'.format(im1.size, im2.size))

    def ssim(self, im1: Image, im2: Image):
        self._image_check(im1=im1, im2=im2)
        if im1.place in (Place.Mat, Place.Ndarray, Place.UMat):
            return self._ssim(im1=im1, im2=im2)
        elif im1.place == Place.GpuMat:
            pass
        else:
            raise ValueError('未知类型必须一致, im1:{}, im2:{}'.format(im1.place, im2.place))

    def _ssim(self, im1: Image, im2: Image):
        h, w = im1.shape[:2]
        new_image_args = {'place': im1.place, 'dtype': self.dtype, 'clone': False}
        nch = im1.channels

        if nch > 1:
            im1 = im1.split()
            im2 = im2.split()
            mssim = np.empty(nch, dtype=np.float64)
            for ch in range(nch):
                result = self._ssim(im1=Image(data=im1[ch], **new_image_args),
                                    im2=Image(data=im2[ch], **new_image_args))
                mssim[ch] = result
            return mssim.mean()

        ux = im1.gaussianBlur(**self.gaussian_args).data
        uy = im2.gaussianBlur(**self.gaussian_args).data

        uxx = Image(data=multiply(im1.data, im1.data), **new_image_args).gaussianBlur(**self.gaussian_args).data
        uyy = Image(data=multiply(im2.data, im2.data), **new_image_args).gaussianBlur(**self.gaussian_args).data
        uxy = Image(data=multiply(im1.data, im2.data), **new_image_args).gaussianBlur(**self.gaussian_args).data

        vx = multiply(self.cov_norm, subtract(uxx, multiply(ux, ux)))
        vy = multiply(self.cov_norm, subtract(uyy, multiply(uy, uy)))
        vxy = multiply(self.cov_norm, subtract(uxy, multiply(ux, uy)))

        C1 = (self.K1 * self.data_range) ** 2
        C2 = (self.K2 * self.data_range) ** 2

        A1 = add(multiply(2, multiply(ux, uy)), C1)
        A2 = add(multiply(2, vxy), C2)
        B1 = add(add(pow(ux, 2), pow(uy, 2)), C1)
        B2 = add(add(vx, vy), C2)

        D = multiply(B1, B2)
        S = divide(multiply(A1, A2), D)

        pad = (self.win_size - 1) // 2
        r = Rect(pad, pad, (w - (2 * pad)), (h - (2 * pad)))

        m = Image(data=S, dtype=np.float64, place=Place.Ndarray).crop(r).data
        m = cv2.mean(m)[0]
        return m
