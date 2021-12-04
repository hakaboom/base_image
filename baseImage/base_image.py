#! usr/bin/python
# -*- coding:utf-8 -*-
import cv2
import base64

from .coordinate import Rect
from .utils import read_image, bytes_2_img, auto_increment
from .exceptions import NoImageDataError, WriteImageError, TransformError
import numpy as np


class _image(object):
    def __init__(self, img=None, flags=cv2.IMREAD_COLOR, path=''):
        """
        基础构造函数
        :param img: 图片数据
        :param flags: 写入图片的cv flags
        :param path: 默认的图片路径, 在读取和写入图片是起到作用
        :return: None
        """
        self.tmp_path = path
        self.image_data = None
        if img is not None:
            self.imwrite(img, flags)

    def save2path(self, path=None):
        """
        写入图片到文件
        :param path: 写入的文件路径
        :return: None
        """
        path = path or self.path
        cv2.imwrite(path, self.imread())

    def imwrite(self, img, flags: int = cv2.IMREAD_COLOR):
        """
        往缓存中写入图片数据
        :param img: 写入的图片数据,可以是图片路径/bytes/numpy.ndarray/cuda_GpuMat/IMAGE
        :param flags: 写入图片的cv flags
        :return: None
        """
        if isinstance(img, str):
            self.image_data = read_image('{}{}'.format(self.tmp_path, img), flags)
        elif isinstance(img, bytes):
            self.image_data = bytes_2_img(img)
        elif isinstance(img, np.ndarray):
            self.image_data = img.copy()
        elif isinstance(img, cv2.cuda_GpuMat):
            self.image_data = img.clone()
        elif isinstance(img, _image):
            raise TypeError('Please use the clone function')
        else:
            raise WriteImageError('Unknown params, type:{}, img={} '.format(type(img), img))

    def imread(self) -> np.ndarray:
        """
        读取图片数据 (内部会自动转换为cpu格式)
        :return: 图片数据(type: numpy.ndarray)
        """
        if self.image_data is not None:
            if self.type == 'cpu':
                return self.image_data
            else:
                return self.image_data.download()
        else:
            raise NoImageDataError('No Image Data in variable')

    def download(self) -> cv2.cuda_GpuMat:
        """
        读取图片数据 (内部会自动转换为gpu格式)
        :return: 图片数据(type: cuda_GpuMat)
        """
        if self.image_data is not None:
            if self.type == 'gpu':
                return self.image_data
            else:
                img = cv2.cuda_GpuMat()
                img.upload(self.imread())
                return img
        else:
            raise NoImageDataError('No Image Data in variable')

    def clean_image(self):
        """
        清除缓存
        :return: None
        """
        self.image_data = None

    @property
    def shape(self) -> tuple:
        """
        获取图片的行、宽、通道数
        :return: 行、宽、通道数
        """
        if self.type == 'cpu':
            return self.imread().shape
        else:
            return self.download().size()[::-1] + (self.download().channels(),)

    @property
    def size(self) -> tuple:
        """
        获取图片的行、宽
        :return: 行、宽
        """
        if self.type == 'cpu':
            return self.imread().shape[:-1]
        else:
            return self.download().size()[::-1]

    def clone(self):
        """
        返回一份copy的IMAGE
        :return: IMAGE
        """
        if self.type == 'cpu':
            return Image(self.imread(), self.path)
        else:
            return Image(self.download(), self.path)

    @property
    def path(self):
        """
        获取图片的默认存放路径
        :return: tmp_path
        """
        return self.tmp_path

    def transform_gpu(self):
        """
        将图片数据转换为cuda_GpuMat
        :return: None
        """
        img = self.image_data
        if isinstance(img, np.ndarray):
            img = cv2.cuda_GpuMat()
            img.upload(self.imread())
            self.imwrite(img)
        elif isinstance(img, cv2.cuda_GpuMat):
            pass
        else:
            raise TransformError('transform Error, img type={}'.format(type(img)))

    def transform_cpu(self):
        """
        将图片数据转换为numpy.ndarray
        :return: None
        """
        img = self.image_data
        if isinstance(img, cv2.cuda_GpuMat):
            img = img.download()
            self.imwrite(img)
        elif isinstance(img, np.ndarray):
            pass
        else:
            raise TransformError('transform Error, img type={}'.format(type(img)))

    @property
    def type(self):
        """
        获取图片数据的类型
        :return: 'cpu'/'gpu'
        """
        if isinstance(self.image_data, np.ndarray):
            return 'cpu'
        elif isinstance(self.image_data, cv2.cuda_GpuMat):
            return 'gpu'


class Image(_image):
    SHOW_INDEX = auto_increment()

    def imshow(self, title: str = None):
        """
        以GUI显示图片
        :param title: cv窗口的名称, 不填写会自动分配
        :return: None
        """
        title = str(title or self.SHOW_INDEX())
        cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(title, self.imread())

    def rotate(self, angle: int = 90, clockwise: bool = True):
        """
        旋转图片
        :param angle: 旋转角度, 默认为90
        :param clockwise: True-顺时针旋转, False-逆时针旋转
        :return: self
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
        """
        区域范围截图,并将截取的区域构建新的IMAGE
        :param rect: 需要截图的范围,可以是Rect/[x,y,width,height]/(x,y,width,height)
        :return: 截取的区域
        """
        img = self.imread()
        height, width = self.size
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
        return Image(img[y_min:y_max, x_min:x_max])

    def binarization(self):
        """
        使用大津法将图片二值化,并返回新的IMAGE
        :return: new IMAGE
        """
        gray_img = self.cvtColor(dst=cv2.COLOR_BGR2GRAY)
        if self.type == 'cpu':
            retval, dst = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
            return Image(dst)
        else:
            # cuda.threshold 不支持大津法
            retval, dst = cv2.threshold(gray_img.download(), 0, 255, cv2.THRESH_OTSU)
            img = cv2.cuda_GpuMat()
            img.upload(dst)
            return Image(img)

    def rectangle(self, rect: Rect):
        """
        在图像上画出矩形
        :param rect: 需要截图的范围,可以是Rect/[x,y,width,height]/(x,y,width,height)
        :return: None
        """
        pt1 = rect.tl
        pt2 = rect.br
        cv2.rectangle(self.imread(), (pt1.x, pt1.y), (pt2.x, pt2.y), (0, 255, 0), 2)

    def resize(self, w, h):
        """
        调整图片大小
        :param w: 需要设定的宽
        :param h: 需要设定的厂
        :return: self
        """
        if self.type == 'cpu':
            img = cv2.resize(self.imread(), (int(w), int(h)))
        else:
            img = cv2.cuda.resize(self.download(), (int(w), int(h)))
        self.imwrite(img)
        return self

    def cv2_to_base64(self) -> bytes:
        """
        将图片数据转换为base64格式
        :return: base64格式的图片数据
        """
        data = cv2.imencode('.png', self.imread())[1].tobytes()
        data = base64.b64encode(data)
        return data

    def cvtColor(self, dst):
        """
        转换图片颜色空间
        :param dst: Destination image
        :return: cuda_GpuMat/numpy.ndarry
        """
        if self.type == 'cpu':
            return cv2.cvtColor(self.imread(), dst)
        else:
            return cv2.cuda.cvtColor(self.download(), dst)

    def rgb_2_gray(self):
        return self.cvtColor(cv2.COLOR_BGR2GRAY)