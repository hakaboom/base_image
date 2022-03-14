import cv2

import numpy as np

from baseImage import Image, Rect
from baseImage.constant import Place


def ssim(im1: Image, im2: Image, win_size: int = 11, data_range: int = 255, sigma: int = 1.5,
         use_sample_covariance=True):
    place = im1.place
    dtype = im1.dtype
    h, w = im1.shape[:2]
    new_img_args = {'place': place, 'dtype': dtype}

    nch = im1.channels

    if nch > 1:
        im1 = im1.split()
        im2 = im2.split()
        mssim = np.empty(nch, dtype=np.float64)

        for ch in range(nch):
            result = ssim(im1=Image(im1[ch], **new_img_args), im2=Image(im2[ch], **new_img_args),
                             win_size=win_size, data_range=data_range, sigma=sigma)
            mssim[ch] = result
        mssim = mssim.mean()
        return mssim

    K1 = 0.01
    K2 = 0.03

    im1.dtype_convert(np.float32)
    im2.dtype_convert(np.float32)

    ndim = 2
    NP = win_size ** ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    gaussian_args = {'size': (win_size, win_size), 'sigma': sigma, 'borderType': cv2.BORDER_REFLECT}

    ux = im1.gaussianBlur(**gaussian_args).data
    uy = im2.gaussianBlur(**gaussian_args).data

    R = data_range

    uxx = Image(data=(im1.data * im1.data), **new_img_args).gaussianBlur(**gaussian_args).data
    uyy = Image(data=(im2.data * im2.data), **new_img_args).gaussianBlur(**gaussian_args).data
    uxy = Image(data=(im1.data * im2.data), **new_img_args).gaussianBlur(**gaussian_args).data

    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D
    pad = (win_size - 1) // 2
    r = Rect(pad, pad, (w - (2 * pad)), (h - (2 * pad)))

    m = Image(data=S, **new_img_args).crop(r).data
    m = cv2.mean(m)[0]

    return m

