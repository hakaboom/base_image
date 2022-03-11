# -*- coding: utf-8 -*-
import cv2
import base64
import paddle
import numpy as np
from loguru import logger

from typing import Tuple, Union, Any
from functools import singledispatchmethod

from .constant import Place, SHOW_INDEX
from .coordinate import Rect, Size
from .utils import read_image, bytes_2_img, auto_increment, cvType_to_npType, npType_to_cvType
from .exceptions import NoImageDataError, WriteImageError, TransformError


try:
    cv2.cuda.GpuMat()
except AttributeError:
    cv2.cuda.GpuMat = cv2.cuda_GpuMat


class _Image(object):

    def __init__(self, data: Union[str, bytes, np.ndarray, cv2.cuda.GpuMat, cv2.Mat, cv2.UMat],
                 read_mode: int = cv2.IMREAD_COLOR, path: str = None,
                 dtype=np.uint8, place=Place.Mat):
        """
        基础构造函数

        Args:
            data(str|bytes|np.ndarray|cv2.cuda.GpuMat): 图片数据
            read_mode(int): 写入图片的cv flags
            path(str): 默认的图片路径, 在读取和写入图片是起到作用
            dtype: 数据格式
            place: 数据存放的方式(np.ndarray|cv2.cuda.GpuMat)

        Returns:
             None
        """
        self._path = path
        self._data = data
        self._read_mode = read_mode
        self._dtype = dtype
        self._place = place

        if data is not None:
            self.write(data, read_mode=self._read_mode, dtype=self.dtype, place=self._place)

    def write(self, data: Union[str, bytes, np.ndarray, cv2.cuda.GpuMat, cv2.Mat, cv2.UMat],
              read_mode: int = None, dtype=None, place=None):
        """
        写入图片数据

        Args:
            data(str|bytes|np.ndarray|cv2.cuda.GpuMat): 图片数据
            read_mode(int): 写入图片的cv flags
            dtype: 数据格式(np.float|np.uint8|...)
            place: 数据存放的方式(np.ndarray|cv2.cuda.GpuMat)

        Returns:
             None

        """
        read_mode = read_mode or self._read_mode
        dtype = dtype or self.dtype
        place = place or self._place

        # logger.debug(f'输入type={type(data)}, id={id(data)}, place={place}')

        if isinstance(data, (str, bytes)):  # data: np.ndarray
            if isinstance(data, str):
                data = read_image(data, flags=read_mode)
            elif isinstance(data, bytes):
                data = bytes_2_img(data)
        elif isinstance(data, np.ndarray):
            data = data.copy()
        elif isinstance(data, cv2.cuda.GpuMat):
            data = data.clone()
        elif isinstance(data, cv2.UMat):
            data = cv2.UMat(data)

        self._data = self.dtype_convert(data, dtype=dtype)
        self._data = self.place_convert(data, place=place)

        # logger.debug(f'输出type={type(self._data)}, id={id(self._data)}, place={place}')

    @classmethod
    def dtype_convert(cls, data: Union[np.ndarray, cv2.cuda.GpuMat, cv2.Mat, cv2.UMat], dtype):
        """
        图片数据类型转换

        Args:
            data: 图片数据
            dtype: 目标数据类型

        Returns:
            data(np.ndarray, cv2.cuda.GpuMat): 图片数据
        """

        if isinstance(data, (np.ndarray, cv2.Mat)):
            if data.dtype != dtype:
                data = data.astype(dtype=dtype)

        elif isinstance(data, cv2.UMat):
            data: np.ndarray = data.get()
            if data.dtype != dtype:
                data = data.astype(dtype=dtype)

        elif isinstance(data, cv2.cuda.GpuMat):
            data_type = cvType_to_npType(data.type(), channel=data.channels())
            if data_type != dtype:
                cvType = npType_to_cvType(dtype, data.channels())
                mat = cv2.cuda.GpuMat(data.size(), cvType)
                data.convertTo(cvType, mat)
                data = mat
        else:
            raise ValueError('Unknown data, type:{}, data={} '.format(type(data), data))

        return data

    @classmethod
    def place_convert(cls, data: Union[np.ndarray, cv2.cuda.GpuMat, cv2.Mat, cv2.UMat], place):
        """
        图片数据格式转换

        Args:
            data: 图片数据
            place: 目标数据格式

        Returns:
            data: 图片数据
        """
        if place in (Place.Ndarray, Place.Mat):
            if isinstance(data, (np.ndarray, cv2.Mat)):
                pass
            elif isinstance(data, cv2.cuda.GpuMat):
                data = data.download()
            elif isinstance(data, cv2.UMat):
                data = data.get()

        elif place == Place.GpuMat:
            if isinstance(data, (np.ndarray, cv2.Mat, cv2.UMat)):
                if type(data) == cv2.UMat:
                    data = data.get()
                mat = cv2.cuda.GpuMat(data.shape[::-1][1:])
                mat.upload(data)
                data = mat
            elif isinstance(data, cv2.cuda.GpuMat):
                pass

        elif place == Place.UMat:
            if isinstance(data, (np.ndarray, cv2.Mat)):
                data = cv2.UMat(data)
            elif isinstance(data, cv2.cuda.GpuMat):
                data = cv2.UMat(data.download())
            elif isinstance(data, cv2.UMat):
                pass
        else:
            raise ValueError('Unknown data, type:{}, data={} '.format(type(data), data))
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
        elif self._place == Place.UMat:
            return self.data.get().shape

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
        elif self._place == Place.UMat:
            return self.data.get().shape[:-1]

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
        elif self._place == Place.UMat:
            return self.data.get().shape[2]

    @property
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


class Image(_Image):
    def clone(self):
        """
        拷贝一个新图片对象

        Returns:
            data: 新图片对象
        """
        return Image(data=self._data, read_mode=self._read_mode, path=self._path,
                     dtype=self.dtype, place=self._place)

    def _clone_with_params(self, data):
        """
        拷贝一个新图片对象

        Returns:
            data: 新图片对象
        """
        return Image(data=data, read_mode=self._read_mode, path=self._path,
                     dtype=self.dtype, place=self._place)

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
        if self._place in (Place.Mat, Place.Ndarray, Place.UMat):
            data = cv2.resize(self.data, (int(w), int(h)))
        elif self._place == Place.GpuMat:
            data = cv2.cuda.resize(self.data, (int(w), int(h)))
        else:
            raise ValueError()

        return self._clone_with_params(data)

    @resize.register(Size)
    def _(self, size: Size):
        """
        调整图片大小

        Args:
            size: 需要设置的长宽

        Returns:
            Image: 调整大小后的图像
        """
        if self._place in (Place.Mat, Place.Ndarray,  Place.UMat):
            data = cv2.resize(self.data, (int(size.width), int(size.height)))
        elif self._place == Place.GpuMat:
            data = cv2.cuda.resize(self.data, (int(size.width), int(size.height)))
        else:
            raise ValueError()

        return self._clone_with_params(data)

    def cvtColor(self, code):
        """
        转换图片颜色空间

        Args:
            code(int): 颜色转换代码
            https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0

        Returns:
            Image: 转换后的新图片
        """
        if self._place in (Place.Mat, Place.Ndarray, Place.UMat):
            data = cv2.cvtColor(self.data, code)
        elif self._place == Place.GpuMat:
            data = cv2.cuda.cvtColor(self.data, code)
        else:
            raise ValueError()

        return self._clone_with_params(data)

    def crop(self, rect: Rect):
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
            data = cv2.cuda.GpuMat(self.data, rect.totuple())
        elif self._place == Place.UMat:
            data = cv2.UMat(self.data, rect.totuple())

        else:
            raise ValueError()

        return self._clone_with_params(data)

    def threshold(self, code=cv2.THRESH_OTSU):
        """
        图片二值化

        Args:
            code: type of the threshold operation

        Returns:
             Image: 二值化后的图片
        """
        if self._place in (Place.Mat, Place.Ndarray, Place.UMat):
            retval, data = cv2.threshold(self.data, 0, 255, code)
        elif self._place == Place.GpuMat:
            retval, data = cv2.threshold(self.data.download(), 0, 255, code)
        else:
            raise ValueError()

        return self._clone_with_params(data)

    def imshow(self, title: str = None, flag: int = cv2.WINDOW_KEEPRATIO):
        """
        以GUI显示图片

        Args:
            title(str): cv窗口的名称, 不填写会自动分配
            flag(int): 窗口类型

        Returns:
            None
        """
        title = str(title or SHOW_INDEX())
        cv2.namedWindow(title, flag)

        if self._place in (Place.Mat, Place.Ndarray, Place.UMat):
            cv2.imshow(title, self.data)
        elif self._place == Place.GpuMat:
            cv2.imshow(title, self.data.download())

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

        if self._place in (Place.Mat, Place.Ndarray, Place.UMat):
            return cv2.rectangle(self.data, (pt1.x, pt1.y), (pt2.x, pt2.y), color, thickness)
        elif self._place == Place.GpuMat:
            cv2.rectangle(self.data, (pt1.x, pt1.y), (pt2.x, pt2.y), color, thickness)
            # np_img = cv2.rectangle(self.data, (pt1.x, pt1.y), (pt2.x, pt2.y), color, thickness)

    # def imread(img_path) -> paddle.Tensor:
    #     img = Image(img_path, flags=cv2.IMREAD_UNCHANGED).imread()
    #     return paddle.to_tensor(img.transpose(2, 0, 1)[None, ...], dtype=paddle.float32)
    #
    #
    # img1 = imread('./image/0.png')
    # img2 = imread('./image/0.png')
    #
    # ssim(img1, img2, data_range=255)
