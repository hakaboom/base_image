#! usr/bin/python
# -*- coding:utf-8 -*-
import cv2
from coordinate import Rect, Size
from utils import read_image, bgr_2_gray, bytes_2_img, auto_increment
import numpy as np


class _image(object):
    def __init__(self, img=None, flags=cv2.IMREAD_COLOR, path=''):
        # 修改掉
        self.tmp_path = path
        self.image_data = None
        if img is not None:
            self.imwrite(img, flags)

    def save2path(self, path=None):
        if self.imread() is None:
            raise ValueError('没有缓存图片')
        path = path or self.path
        cv2.imwrite(path, self.imread())

    def imwrite(self, img, flags: int = cv2.IMREAD_COLOR):
        if type(img) == str:
            self.image_data = read_image('{}{}'.format(self.tmp_path, img), flags)
        elif isinstance(img, np.ndarray) or isinstance(img, cv2.cuda_GpuMat):
            self.image_data = img
        elif isinstance(img, _image):
            self.image_data = img.imread().copy()
        elif isinstance(img, bytes):
            self.image_data = bytes_2_img(img)
        else:
            raise ValueError('unknown image, type:{}, image={} '.format(type(img), img))

    def imread(self) -> np.ndarray:
        self.transform_cpu()
        return self.image_data

    def download(self) -> cv2.cuda_GpuMat:
        self.transform_gpu()
        return self.image_data

    def clean_image(self):
        """清除缓存"""
        self.image_data = None

    @property
    def shape(self):
        if self.type() == 'cpu':
            return self.imread().shape[:-1]
        elif self.type() == 'gpu':
            return self.download().size()[::-1]

    @property
    def path(self):
        return self.tmp_path

    def transform_gpu(self):
        img = self.image_data
        if isinstance(img, np.ndarray):
            img = cv2.cuda_GpuMat()
            img.upload(self.imread())
            self.imwrite(img)
        elif isinstance(img, cv2.cuda_GpuMat):
            pass
        else:
            raise TypeError('transform Error, img type={}'.format(type(img)))

    def transform_cpu(self):
        img = self.image_data
        if isinstance(img, cv2.cuda_GpuMat):
            img = img.download()
            self.imwrite(img)
        elif isinstance(img, np.ndarray):
            pass
        else:
            raise TypeError('transform Error, img type={}'.format(type(img)))

    def type(self):
        if isinstance(self.image_data, np.ndarray):
            return 'cpu'
        elif isinstance(self.image_data, cv2.cuda_GpuMat):
            return 'gpu'


class IMAGE(_image):
    SHOW_INDEX = auto_increment()

    def imshow(self, title: str = None):
        """以GUI显示图片"""
        title = str(title or self.SHOW_INDEX())
        cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(title, self.imread())
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def rotate(self, angle: int = 90, clockwise: bool = True):
        """
        旋转图片

        Args:
            angle: 旋转角度
            clockwise: True-顺时针旋转, False-逆时针旋转
        """
        img = self.imread()
        if clockwise:
            angle = 360 - angle
        rows, cols, _ = img.shape
        center = (cols / 2, rows / 2)
        mask = img.copy()
        mask[:, :] = 255
        M = cv2.getRotationMatrix2D(center, angle, 1)
        top_right = np.array((cols, 0)) - np.array(center)
        bottom_right = np.array((cols, rows)) - np.array(center)
        top_right_after_rot = M[0:2, 0:2].dot(top_right)
        bottom_right_after_rot = M[0:2, 0:2].dot(bottom_right)
        new_width = max(int(abs(bottom_right_after_rot[0] * 2) + 0.5), int(abs(top_right_after_rot[0] * 2) + 0.5))
        new_height = max(int(abs(top_right_after_rot[1] * 2) + 0.5), int(abs(bottom_right_after_rot[1] * 2) + 0.5))
        offset_x, offset_y = (new_width - cols) / 2, (new_height - rows) / 2
        M[0, 2] += offset_x
        M[1, 2] += offset_y
        self.imwrite(cv2.warpAffine(img, M, (new_width, new_height)))
        return self

    def crop_image(self, rect):
        """区域范围截图"""
        img = self.imread()
        height, width = self.shape
        if isinstance(rect, (list, tuple)) and len(rect) == 4:
            rect = Rect(*rect)
        elif isinstance(rect, Rect):
            pass
        else:
            raise ValueError('unknown rect: type={}, rect={}'.format(type(rect), rect))
        if not Rect(0, 0, width, height).contains(rect):
            raise OverflowError('Rect不能超出屏幕 rect={}, tl={}, br={}'.format(rect, rect.tl, rect.br))
        # 获取在图像中的实际有效区域：
        x_min, y_min = int(rect.tl.x), int(rect.tl.y)
        x_max, y_max = int(rect.br.x), int(rect.br.y)
        return IMAGE(img[y_min:y_max, x_min:x_max])

    def binarization(self):
        gray_img = self.rgb_2_gray()
        dst = None
        if self.type() == 'cpu':
            retval, dst = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
        elif self.type() == 'gpu':
            retval, dst = cv2.threshold(gray_img.download(), 0, 255, cv2.THRESH_OTSU)
        return IMAGE(dst)

    def rectangle(self, rect: Rect):
        """在图像上画出矩形"""
        pt1 = rect.tl
        pt2 = rect.br
        cv2.rectangle(self.imread(), (pt1.x, pt1.y), (pt2.x, pt2.y), (0, 255, 0), 2)

    def resize(self, w, h):
        img = None
        if self.type() == 'cpu':
            img = cv2.resize(self.imread(), (int(w), int(h)))
        elif self.type() == 'gpu':
            img = cv2.cuda.resize(self.download(), (int(w), int(h)))
        self.imwrite(img)
        return self

    def cv2_to_base64(self):
        data = cv2.imencode('.png', self.imread())
        return data

    def rgb_2_gray(self):
        if self.type() == 'cpu':
            return cv2.cvtColor(self.imread(), cv2.COLOR_BGR2GRAY)
        elif self.type() == 'gpu':
            return cv2.cuda.cvtColor(self.download(), cv2.COLOR_BGR2GRAY)


def check_detection_input(im_source, im_search):
    return IMAGE(im_source), IMAGE(im_search)