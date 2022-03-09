#! usr/bin/python
# -*- coding:utf-8 -*-
import cv2
import base64
import numpy as np

from typing import Tuple

from .coordinate import Rect
from .utils import read_image, bytes_2_img, auto_increment
from .exceptions import NoImageDataError, WriteImageError, TransformError


class Image(object):
    SHOW_INDEX = auto_increment()

    def __init__(self, data=None, flags=cv2.IMREAD_COLOR, path=''):
        """
        基础构造函数

        Args:
            data(str|bytes|np.ndarray|cv2.cuda.GpuMat): 图片数据
            flags(int): 写入图片的cv flags
            path(str): 默认的图片路径, 在读取和写入图片是起到作用

        Returns:
             None
        """
        self._path = path
        self.image_data = None
        if data is not None:
            self.imwrite(data, flags)

    def save2path(self, path=None):
        """
        写入图片到文件
        Args:
            path(str): 写入的文件路径

        Returns:
             None
        """
        path = path or self.path
        cv2.imwrite(path, self.imread())

    def imwrite(self, data, flags: int = cv2.IMREAD_COLOR):
        """
        往缓存中写入图片数据

        Args:
            data(str|bytes|np.ndarray|cv2.cuda.GpuMat): 写入的图片数据
            flags(int): 写入图片的cv flags

        Returns:
            None
        """
        if isinstance(data, str):
            self.image_data = read_image('{}{}'.format(self.path, data), flags)
        elif isinstance(data, bytes):
            self.image_data = bytes_2_img(data)
        elif isinstance(data, np.ndarray):
            self.image_data = data.copy()
        elif isinstance(data, cv2.cuda.GpuMat):
            self.image_data = data.clone()
        elif isinstance(data, Image):
            raise TypeError('Please use the clone function')
        else:
            raise WriteImageError('Unknown params, type:{}, data={} '.format(type(data), data))

    def imread(self) -> np.ndarray:
        """
        读取图片数据

        Returns:
            img(np.ndarray): 图片数据
        """
        if self.image_data is not None:
            if self.type == 'cpu':
                return self.image_data
            else:
                return self.image_data.download()
        else:
            raise NoImageDataError('No Image Data in variable')

    def download(self) -> cv2.cuda.GpuMat:
        """
        读取图片数据

        Returns:
            data(cv2.cuda.GpuMat): 图片数据
        """
        if self.image_data is not None:
            if self.type == 'gpu':
                return self.image_data
            else:
                img = cv2.cuda.GpuMat()
                img.upload(self.imread())
                return img
        else:
            raise NoImageDataError('No Image Data in variable')

    def clean_image(self):
        """
        清除缓存

        Returns:
            None
        """
        self.image_data = None

    @property
    def shape(self) -> tuple:
        """
        获取图片的行、宽、通道数

        Returns:
            行、宽、通道数
        """
        if self.type == 'cpu':
            return self.imread().shape
        else:
            return self.download().size()[::-1] + (self.download().channels(),)

    @property
    def size(self) -> tuple:
        """
        获取图片的行、宽

        Returns:
            行、宽
        """
        if self.type == 'cpu':
            return self.imread().shape[:-1]
        else:
            return self.download().size()[::-1]

    def clone(self):
        """
        拷贝一个新的图片对象

        Returns:
            image(Image): 新图片
        """
        if self.type == 'cpu':
            return Image(self.imread(), path=self.path)
        else:
            return Image(self.download(), path=self.path)

    @property
    def path(self) -> str:
        """
        获取图片的默认存放路径

        Returns:
            图片路径
        """
        return self._path

    def transform_gpu(self):
        """
        将图片数据转换为cuda.GpuMat

        Returns:
            None
        """
        img = self.image_data
        if isinstance(img, np.ndarray):
            img = cv2.cuda.GpuMat()
            img.upload(self.imread())
            self.imwrite(img)
        elif isinstance(img, cv2.cuda.GpuMat):
            pass
        else:
            raise TransformError('transform Error, img type={}'.format(type(img)))

    def transform_cpu(self):
        """
        将图片数据转换为numpy.ndarray

        Returns:
            None
        """
        img = self.image_data
        if isinstance(img, cv2.cuda.GpuMat):
            img = img.download()
            self.imwrite(img)
        elif isinstance(img, np.ndarray):
            pass
        else:
            raise TransformError('transform Error, img type={}'.format(type(img)))

    @property
    def type(self) -> str:
        """
        获取图片数据的类型

        Returns:
             type(str): cpu/gpu
        """
        if isinstance(self.image_data, np.ndarray):
            return 'cpu'
        elif isinstance(self.image_data, cv2.cuda.GpuMat):
            return 'gpu'

    def imshow(self, title: str = None):
        """
        以GUI显示图片

        Args:
            title(str): cv窗口的名称, 不填写会自动分配

        Returns:
            None
        """
        title = str(title or self.SHOW_INDEX())
        cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(title, self.imread())

    def rotate(self, angle: int = 90, clockwise: bool = True):
        """
        旋转图片

        Args:
            angle(int): 旋转角度, 默认为90
            clockwise(bool): True-顺时针旋转, False-逆时针旋转

        Returns:
            return self
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

        Args:
            rect: 需要截图的范围,可以是Rect/[x,y,width,height]/(x,y,width,height)

        Returns:
             Image: 截取的区域
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
        使用大津法将图片二值化

        Returns:
            image(Image): 新图片
        """
        gray_img = self.cvtColor(dst=cv2.COLOR_BGR2GRAY)
        if self.type == 'cpu':
            retval, dst = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
            return Image(dst)
        else:
            # cuda.threshold 不支持大津法
            retval, dst = cv2.threshold(gray_img.download(), 0, 255, cv2.THRESH_OTSU)
            img = cv2.cuda.GpuMat()
            img.upload(dst)
            return Image(img)

    def rectangle(self, rect: Rect, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 1):
        """
        在图像上画出矩形

        Args:
            rect(Rect): 需要截图的范围,可以是Rect/[x,y,width,height]/(x,y,width,height)
            color(tuple): 表示矩形边框的颜色
            thickness(int): 形边框的厚度

        Returns:
            None
        """
        pt1 = rect.tl
        pt2 = rect.br
        cv2.rectangle(self.imread(), (pt1.x, pt1.y), (pt2.x, pt2.y), color, thickness)

    def resize(self, w: int, h: int):
        """
        调整图片大小

        Args:
            w(int): 需要设定的宽
            h(int): 需要设定的厂

        Returns:
            self
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

        Returns:
            data(bytes):base64格式的图片数据
        """
        data = cv2.imencode('.png', self.imread())[1].tobytes()
        data = base64.b64encode(data)
        return data

    def cvtColor(self, dst):
        """
        转换图片颜色空间

        Args:
            dst(int): Destination image

        Returns:
            data(cv2.cuda.GpuMat/np.ndarry)
        """
        if self.type == 'cpu':
            return cv2.cvtColor(self.imread(), dst)
        else:
            return cv2.cuda.cvtColor(self.download(), dst)

    def rgb_2_gray(self):
        return self.cvtColor(cv2.COLOR_BGR2GRAY)

    # paddle
    def np2tensor(self):
        """
        转换ndarray成paddle.Tensor

        Returns:
            paddle.Tensor
        """
        return paddle.to_tensor(self.imread().transpose(2, 0, 1)[None, ...], dtype=paddle.float32)

    @staticmethod
    def _fspecial_gauss_1d(size, sigma):
        """
        Create 1-D gauss kernel

        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            paddle.Tensor: 1D kernel (1 x 1 x size)
        """
        coords = paddle.arange(size, dtype=paddle.float32)
        coords -= size // 2

        g = paddle.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()

        return g.unsqueeze(0).unsqueeze(0)

    def gaussian_filter(self, win=None, win_size=11, win_sigma=1.5):
        data = self.np2tensor()

        if win is None:
            win = self._fspecial_gauss_1d(win_size, win_sigma)
            win = win.tile([data.shape[1]] + [1] * (len(data.shape) - 1))

        assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape

        if len(data.shape) != 4:
            raise NotImplementedError(data.shape)

        C = data.shape[1]
        out = data
        for i, s in enumerate(data.shape[2:]):
            if s >= win.shape[-1]:
                perms = list(range(win.ndim))
                perms[2 + i] = perms[-1]
                perms[-1] = 2 + i
                out = paddle.nn.functional.conv2d(out, weight=win.transpose(perms), stride=1, padding=0, groups=C)
            else:
                print(
                    f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {data.shape} and win size: {win.shape[-1]}"
                )

        return out
