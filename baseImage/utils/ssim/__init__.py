from baseImage import Image, Rect
from baseImage.constant import Place

import cv2
import numpy as np
from typing import Union


operations = {
    'mat': {
        'multiply': cv2.multiply,
        'subtract': cv2.subtract,
        'add': cv2.add,
        'pow': cv2.pow,
        'divide': cv2.divide,
    },
    'cuda': {
        'multiply': cv2.cuda.multiply,
        'subtract': cv2.cuda.subtract,
        'add': cv2.cuda.add,
        'pow': cv2.cuda.pow,
        'divide': cv2.cuda.divide,
    }
}


class SSIM(object):
    def __init__(self, win_size: int = 11, data_range: int = 255, sigma: Union[int, float] = 1.5,
                 use_sample_covariance=True, resize=(500, 500)):
        """
        使用ssim计算两张图片的相似度

        Args:
            win_size(int): 高斯核大小
            data_range(int): 图片的取值范围 (1.0 or 255)
            sigma(float): 高斯核标准差
            use_sample_covariance: 如果为真，则通过 N-1 而不是 N 归一化
            resize: 输入图片的缩放大小
        """
        self.win_size = win_size
        self.data_range = data_range
        self.sigma = sigma
        self.dtype = np.float32
        self.K1 = 0.01
        self.K2 = 0.03
        self.C1 = (self.K1 * self.data_range) ** 2
        self.C2 = (self.K2 * self.data_range) ** 2
        self.resize = resize

        NP = win_size ** 2
        # filter has already normalized by NP
        if use_sample_covariance:
            cov_norm = NP / (NP - 1)  # sample covariance
        else:
            cov_norm = 1.0  # population covariance to match Wang et. al. 2004
        self.cov_norm = cov_norm
        self.gaussian_args = {'size': (win_size, win_size), 'sigma': sigma, 'borderType': cv2.BORDER_REFLECT}

        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self._buffer_cuda_mat()

    def _buffer_cuda_mat(self):
        size = self.resize

        cov_norm = np.empty(size, dtype=np.float32)
        cov_norm.fill(self.cov_norm)
        self._cuda_cov_norm = cv2.cuda.GpuMat(size, cv2.CV_32F)
        self._cuda_cov_norm.upload(cov_norm)

        C1 = np.empty(size, dtype=np.float32)
        C1.fill(self.C1)
        self._cuda_C1 = cv2.cuda.GpuMat(size, cv2.CV_32F)
        self._cuda_C1.upload(C1)

        C2 = np.empty(size, dtype=np.float32)
        C2.fill(self.C2)
        self._cuda_C2 = cv2.cuda.GpuMat(size, cv2.CV_32F)
        self._cuda_C2.upload(C2)

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
        im1 = im1.resize(*self.resize)
        im2 = im2.resize(*self.resize)

        return self._ssim(im1=im1, im2=im2)

    def _ssim(self, im1: Image, im2: Image):
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

        return self._cv_ssim(im1=im1, im2=im2)

    def _cv_ssim(self, im1: Image, im2: Image):
        h, w = im1.shape[:2]
        new_image_args = {'place': im1.place, 'dtype': self.dtype, 'clone': False}
        cuda_flag = im1.place == Place.GpuMat

        ux = im1.gaussianBlur(**self.gaussian_args).data
        uy = im2.gaussianBlur(**self.gaussian_args).data

        if cuda_flag:
            multiply = operations['cuda']['multiply']
            subtract = operations['cuda']['subtract']
            add = operations['cuda']['add']
            divide = operations['cuda']['divide']
            pow = operations['cuda']['pow']

            cov_norm = self._cuda_cov_norm
            C1 = self._cuda_C1
            C2 = self._cuda_C2
        else:
            multiply = operations['mat']['multiply']
            subtract = operations['mat']['subtract']
            add = operations['mat']['add']
            divide = operations['mat']['divide']
            pow = operations['mat']['pow']

            cov_norm = self.cov_norm
            C1 = self.C1
            C2 = self.C2

        uxx = Image(data=multiply(im1.data, im1.data), **new_image_args).gaussianBlur(**self.gaussian_args).data
        uyy = Image(data=multiply(im2.data, im2.data), **new_image_args).gaussianBlur(**self.gaussian_args).data
        uxy = Image(data=multiply(im1.data, im2.data), **new_image_args).gaussianBlur(**self.gaussian_args).data

        vx = multiply(cov_norm, subtract(uxx, multiply(ux, ux)))
        vy = multiply(cov_norm, subtract(uyy, multiply(uy, uy)))
        vxy = multiply(cov_norm, subtract(uxy, multiply(ux, uy)))

        if cuda_flag:
            _A1 = multiply(ux, uy)
            A1 = add(add(_A1, _A1), C1)
            A2 = add(add(vxy, vxy), C2)
            B1 = add(add(pow(ux, 2), pow(uy, 2)), C1)
            B2 = add(add(vx, vy), C2)
            D = multiply(B1, B2)
            S = divide(multiply(A1, A2), D)
        else:
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