from baseImage import Image
from baseImage.constant import Place

import cv2
from .cuda import ssim as cuda_ssim
from .ndarray import ssim as np_ssim
from .umat import ssim as umat_ssim


def ssim(im1: Image, im2: Image, win_size: int = 11, data_range: int = 255, sigma: int = 1.5,
         use_sample_covariance: bool = True):
    if not isinstance(im1, Image) and not isinstance(im2, Image):
        raise ValueError('im1 im2必须为Image类型, im1_type:{}, im2_type:{}'.format(type(im1), type(im2)))

    if im1.place != im2.place:
        raise ValueError('图片类型必须一致, im1:{}, im2:{}'.format(im1.place, im2.place))

    if im1.channels != im2.channels:
        raise ValueError('图片通道数量必须一致, im1:{}, im2:{}'.format(im1.place, im2.place))

    ssim_args = {'im1': im1, 'im2': im2, 'win_size': win_size, 'data_range': data_range, 'sigma': sigma,
                 'use_sample_covariance': use_sample_covariance}
    if im1.place in (Place.Ndarray, Place.Mat):
        return np_ssim(**ssim_args)
    elif im1.place == Place.UMat:
        return umat_ssim(**ssim_args)
    elif im1.place == Place.GpuMat:
        return cuda_ssim(**ssim_args)
    else:
        raise ValueError('未知类型必须一致, im1:{}, im2:{}'.format(im1.place, im2.place))