# -*- coding: utf-8 -*-
import cv2
import base64
import paddle
import numpy as np

from typing import Tuple, Union, Any
from functools import singledispatchmethod

from .constant import Place
from .coordinate import Rect, Size
from .utils import read_image, bytes_2_img, auto_increment, cvType_to_npType
from .exceptions import NoImageDataError, WriteImageError, TransformError


class _Image(object):

    def __init__(self, data: Union[str, bytes, np.ndarray, cv2.cuda_GpuMat, cv2.Mat],
                 read_mode: int = cv2.IMREAD_COLOR, path: str = None,
                 dtype=np.uint8, place=Place.Mat):
        """
        基础构造函数

        Args:
            data(str|bytes|np.ndarray|cv2.cuda_GpuMat): 图片数据
            read_mode(int): 写入图片的cv flags
            path(str): 默认的图片路径, 在读取和写入图片是起到作用
            dtype: 数据格式
            place: 数据存放的方式(np.ndarray|cv2.cuda_GpuMat)

        Returns:
             None
        """
        self._path = path
        self._data = data
        self._read_mode = read_mode
        self._dtype = dtype
        self._place = place

        if data is not None:
            self.write(data, read_mode=self._read_mode, dtype=self._dtype, place=self._place)

    def write(self, data: Union[str, bytes, np.ndarray, cv2.cuda_GpuMat, cv2.Mat],
              read_mode: int = None, dtype=None, place=None):
        """
        写入图片数据

        Args:
            data(str|bytes|np.ndarray|cv2.cuda_GpuMat): 图片数据
            read_mode(int): 写入图片的cv flags
            dtype: 数据格式(np.float|np.uint8|...)
            place: 数据存放的方式(np.ndarray|cv2.cuda_GpuMat)

        Returns:
             None

        """
        read_mode = read_mode or self._read_mode
        dtype = dtype or self._dtype
        place = place or self._place

        if isinstance(data, (np.ndarray, str, bytes)):  # data: np.ndarray
            if isinstance(data, str):
                data = read_image(data, flags=read_mode)
            elif isinstance(data, bytes):
                data = bytes_2_img(data)
            else:
                data = data.copy()

        elif isinstance(data, cv2.cuda_GpuMat):
            data = data.clone()
        else:
            raise ValueError('Unknown data, type:{}, data={} '.format(type(data), data))

        data = self.dtype_convert(data, dtype=dtype)
        data = self.place_convert(data, place=place)

        self._data = data

    @singledispatchmethod
    @classmethod
    def dtype_convert(cls, data: Union[np.ndarray, cv2.cuda_GpuMat, cv2.Mat], dtype):
        """
        图片数据类型转换

        Args:
            data: 图片数据
            dtype: 目标数据类型

        Returns:
            data(np.ndarray, cv2.cuda_GpuMat): 图片数据
        """
        raise ValueError('Unknown data, type:{}, data={} '.format(type(data), data))
    
    @dtype_convert.register(np.ndarray)
    @dtype_convert.register(cv2.Mat)
    @classmethod
    def _(cls, data, dtype):
        if data.dtype != dtype:
            data = data.astype(dtype)
        return data

    @dtype_convert.register(cv2.cuda_GpuMat)
    @classmethod
    def _(cls, data, dtype):
        data_type = cvType_to_npType(data.type(), channel=data.channels())
        if data_type != dtype:
            data.upload(data.download().astype(dtype=dtype))
        return data

    @singledispatchmethod
    @classmethod
    def place_convert(cls, data: Union[np.ndarray, cv2.cuda_GpuMat, cv2.Mat], place):
        """
        图片数据格式转换

        Args:
            data: 图片数据
            place: 目标数据格式

        Returns:
            data: 图片数据
        """
        raise ValueError('Unknown data, type:{}, data={} '.format(type(data), data))

    @place_convert.register(np.ndarray)
    @place_convert.register(cv2.Mat)
    @classmethod
    def _(cls, data, place):
        if place in (Place.Ndarray, Place.Mat):
            pass
        elif place == Place.GpuMat:
            mat = cv2.cuda_GpuMat()
            # TODO: 不支持的dtype
            mat.upload(data)
            data = mat
        return data

    @place_convert.register(cv2.cuda_GpuMat)
    @classmethod
    def _(cls, data, place):
        if place in (Place.Ndarray, Place.Mat):
            data = data.download()
        elif place == Place.GpuMat:
            pass
        return data

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        获取图片的长、宽、通道数

        Returns:
            shape: (长,宽,通道数)
        """
        if self._place in (Place.Mat, Place.Ndarray):
            return self.data.shape
        elif self._place == Place.GpuMat:
            return self.data.size()[::-1] + (self.data.channels(),)

    @property
    def size(self):
        """
        获取图片的长、宽

        Returns:
            shape: (长,宽)
        """
        if self._place in (Place.Mat, Place.Ndarray):
            return self.data.shape[:-1]
        elif self._place == Place.GpuMat:
            return self.data.size()[::-1]

    @property
    def channels(self):
        """
        获取图片的通道数

        Returns:
            channels: 通道数
        """
        if self._place in (Place.Mat, Place.Ndarray):
            return self.data.shape[2]
        elif self._place == Place.GpuMat:
            return self.data.channels()

    def dtype(self):
        """
        获取图片的数据类型

        Returns:
            dtype: 数据类型
        """
        return self._dtype

    @property
    def data(self):
        return self._data

    def clone(self):
        """
        拷贝一个新图片对象

        Returns:
            data: 新图片对象
        """
        return _Image(data=self._data, read_mode=self._read_mode, path=self._path,
                      dtype=self._dtype, place=self._place)

    def _clone_with_params(self, data):
        """
        拷贝一个新图片对象

        Returns:
            data: 新图片对象
        """
        return _Image(data=data, read_mode=self._read_mode, path=self._path,
                      dtype=self._dtype, place=self._place)


class Image(_Image):
    @singledispatchmethod
    def resize(self, w: int, h: int):
        """
        调整图片大小

        Args:
            w(int): 需要设定的宽
            h(int): 需要设定的长

        Returns:
            Image: 调整大小后的图像
        """
        if self._place in (Place.Mat, Place.Ndarray):
            data = cv2.resize(self.data, (int(w), int(h)))
        elif self._place == Place.GpuMat:
            data = cv2.cuda.resize(self.data, (int(w), int(h)))

        return self._clone_with_params(data)

    @resize.register(Size)
    def _(self, size: Size) -> _Image:
        """
        调整图片大小

        Args:
            size: 需要设置的长宽

        Returns:
            Image: 调整大小后的图像
        """
        if self._place in (Place.Mat, Place.Ndarray):
            data = cv2.resize(self.data, (int(size.width), int(size.height)))
        elif self._place == Place.GpuMat:
            data = cv2.cuda.resize(self.data, (int(size.width), int(size.height)))

        return self._clone_with_params(data)

    def cvtColor(self, code) -> _Image:
        """
        转换图片颜色空间

        Args:
            code(int): 颜色转换代码
            https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0

        Returns:
            Image: 转换后的新图片
        """
        if self._place in (Place.Mat, Place.Ndarray):
            data = cv2.cvtColor(self.data, code)
        elif self._place == Place.GpuMat:
            data = cv2.cuda.cvtColor(self.data, code)

        return self._clone_with_params(data)

    def crop(self, rect: Rect) -> _Image:
        """
        区域范围截图

        Args:
            rect: 需要截图的范围

        Returns:
             Image: 截取的区域
        """
        height, width = self.size
        if not Rect(0, 0, width, height).contains(rect):
            raise OverflowError('Rect不能超出屏幕 rect={}, tl={}, br={}'.format(rect, rect.tl, rect.br))

        if self._place in (Place.Mat, Place.Ndarray):
            x_min, y_min = int(rect.tl.x), int(rect.tl.y)
            x_max, y_max = int(rect.br.x), int(rect.br.y)
            data = self._data[y_min:y_max, x_min:x_max]
        elif self._place == Place.GpuMat:
            data = cv2.cuda_GpuMat(self.data, rect.totuple())

        return self._clone_with_params(data)

    # def imread(img_path) -> paddle.Tensor:
    #     img = Image(img_path, flags=cv2.IMREAD_UNCHANGED).imread()
    #     return paddle.to_tensor(img.transpose(2, 0, 1)[None, ...], dtype=paddle.float32)
    #
    #
    # img1 = imread('./image/0.png')
    # img2 = imread('./image/0.png')
    #
    # ssim(img1, img2, data_range=255)
