# -*- coding: utf-8 -*-
import cv2
import numpy as np
from loguru import logger

from typing import Tuple, Union
from functools import singledispatchmethod

from .constant import Place, SHOW_INDEX
from .coordinate import Rect, Size
from .utils import read_image, bytes_2_img, cvType_to_npType, npType_to_cvType
from .exceptions import NoImageDataError, WriteImageError, TransformError


try:
    cv2.cuda.GpuMat()
except AttributeError:
    cv2.cuda.GpuMat = cv2.cuda_GpuMat


class _Image(object):

    def __init__(self, data: Union[str, bytes, np.ndarray, cv2.cuda.GpuMat, cv2.Mat, cv2.UMat],
                 read_mode: int = cv2.IMREAD_COLOR, dtype=np.uint8, place=Place.Mat, clone: bool = True):
        """
        基础构造函数

        Args:
            data(str|bytes|np.ndarray|cv2.cuda.GpuMat|cv2.UMat): 图片数据
                str: 接收一个文件路径,读取该路径的图片数据,转换为ndarray
                bytes: 接收bytes,转换为ndarray
                cv2.Mat|cv2.UMat|cv2.cuda.GpuMat: 接收opencv的图片格式

            read_mode(int): 写入图片的cv flags
            dtype: 数据格式
            place: 数据存放的方式(np.ndarray|cv2.cuda.GpuMat)
            clone(bool): if True图片数据会被copy一份新的, if False则不会拷贝

        Returns:
             None
        """
        self._data = data
        self._read_mode = read_mode
        self._dtype = dtype
        self._place = place

        if data is not None:
            self.write(data, read_mode=self._read_mode, dtype=self.dtype, place=self._place, clone=clone)

    def write(self, data: Union[str, bytes, np.ndarray, cv2.cuda.GpuMat, cv2.Mat, cv2.UMat],
              read_mode: int = None, dtype=None, place=None, clone=True):
        """
        写入图片数据

        Args:
            data(str|bytes|np.ndarray|cv2.cuda.GpuMat): 图片数据
            read_mode(int): 写入图片的cv flags
            dtype: 数据格式(np.float|np.uint8|...)
            place: 数据存放的方式(np.ndarray|cv2.cuda.GpuMat)
            clone(bool): if True图片数据会被copy一份新的, if False则不会拷贝

        Returns:
             None

        """
        read_mode = read_mode or self._read_mode
        dtype = dtype or self.dtype
        place = place or self._place

        # logger.debug(f'输入type={type(data)}, id={id(data)}, place={place}')

        if not clone:
            self._data = data
            return

        if isinstance(data, _Image):
            data = data.data

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

        self._data = data
        # 先转换类型,再转换数据格式
        self.place_convert(place=place)
        self.dtype_convert(dtype=dtype)
        # logger.debug(f'输出type={type(self._data)}, id={id(self._data)}, place={place}')

    @classmethod
    def _create_mat(cls, data: Union[np.ndarray, cv2.Mat], shape: Union[tuple, list]):
        if len(shape) == 2:  # 当Mat和Ndarray为单通道时,shape会缺少通道
            shape = shape + (1,)

        if shape[2] == 1:
            mat = cv2.Mat(data, wrap_channels=False)
        else:
            mat = cv2.Mat(data, wrap_channels=True)
        return mat

    def dtype_convert(self, dtype):
        """
        图片数据类型转换

        Args:
            dtype: 目标数据类型

        Returns:
            data(np.ndarray, cv2.cuda.GpuMat): 图片数据
        """
        data = self._data

        if isinstance(data, cv2.Mat):
            if data.dtype != dtype:
                data = data.astype(dtype=dtype)

        elif isinstance(data, np.ndarray):
            if data.dtype != dtype:
                data = data.astype(dtype=dtype)

        elif isinstance(data, cv2.UMat):
            _data: np.ndarray = data.get()
            if _data.dtype != dtype:
                data = _data.astype(dtype=dtype)
                data = cv2.UMat(data)

        elif isinstance(data, cv2.cuda.GpuMat):
            data_type = cvType_to_npType(data.type(), channels=data.channels())
            if data_type != dtype:
                cvType = npType_to_cvType(dtype, data.channels())
                mat = cv2.cuda.GpuMat(data.size(), cvType)
                data.convertTo(cvType, mat)
                data = mat
        else:
            raise ValueError('Unknown data, type:{}, data={} '.format(type(data), data))

        self._data = data
        self._dtype = dtype

    def place_convert(self, place):
        """
        图片数据格式转换

        Args:
            place: 目标数据格式

        Returns:
            data: 图片数据
        """
        data = self._data

        if place == Place.Ndarray:
            if type(data) == np.ndarray:
                pass
            elif type(data) == cv2.Mat:
                data = np.asarray(data)
            elif isinstance(data, cv2.cuda.GpuMat):
                data = data.download()
            elif isinstance(data, cv2.UMat):
                data = data.get()

        elif place == Place.Mat:
            if type(data) == np.ndarray:
                pass
            elif type(data) == cv2.Mat:
                pass
            elif isinstance(data, cv2.cuda.GpuMat):
                data = data.download()
            elif isinstance(data, cv2.UMat):
                data = data.get()
            data = self._create_mat(data, data.shape)

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

        self._data = data
        self._place = place

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        获取图片的长、宽、通道数

        Returns:
            shape: (长,宽,通道数)
        """
        if self._place in (Place.Mat, Place.Ndarray):
            shape = self.data.shape
        elif self._place == Place.GpuMat:
            shape = self.data.size()[::-1] + (self.data.channels(),)
        elif self._place == Place.UMat:
            shape = self.data.get().shape

        if len(shape) == 2:  # 当Mat和Ndarray为单通道时,shape会缺少通道
            shape = shape + (1,)

        return shape

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
            return self.shape[2]
        elif self._place == Place.GpuMat:
            return self.data.channels()
        elif self._place == Place.UMat:
            return self.shape[2]

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
        return Image(data=self._data, read_mode=self._read_mode, dtype=self.dtype, place=self._place)

    def _clone_with_params(self, data, **kwargs):
        """
        拷贝一个新图片对象

        Returns:
            data: 新图片对象
        """
        clone = kwargs.pop('clone', True)
        return Image(data=data, read_mode=self._read_mode, dtype=self.dtype, place=self._place, clone=clone)

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
        if self._place == Place.Mat:
            data = cv2.resize(self.data, (int(w), int(h)))  # return: np.ndarray
            data = self._create_mat(data, data.shape)
        elif self._place in (Place.Ndarray, Place.UMat):
            data = cv2.resize(self.data, (int(w), int(h)))
        elif self._place == Place.GpuMat:
            data = cv2.cuda.resize(self.data, (int(w), int(h)))
        else:
            raise TypeError("Unknown place:'{}', image_data={}, image_data_type".format(self._place, self.data, type(self.data)))
        return self._clone_with_params(data, clone=False)

    @resize.register(Size)
    def _(self, size: Size):
        """
        调整图片大小

        Args:
            size: 需要设置的长宽

        Returns:
            Image: 调整大小后的图像
        """
        return self.resize(int(size.width), int(size.height))

    def cvtColor(self, code):
        """
        转换图片颜色空间

        Args:
            code(int): 颜色转换代码
            https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0

        Returns:
            Image: 转换后的新图片
        """
        if self._place == Place.Mat:
            data = cv2.cvtColor(self.data, code)  # return np.ndarray
            data = self._create_mat(data, shape=data.shape)
        elif self._place in (Place.Ndarray, Place.UMat):
            data = cv2.cvtColor(self.data, code)
        elif self._place == Place.GpuMat:
            data = cv2.cuda.cvtColor(self.data, code)
        else:
            raise TypeError("Unknown place:'{}', image_data={}, image_data_type".format(self._place, self.data, type(self.data)))

        return self._clone_with_params(data, clone=False)

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
            data = self.data.copy()[y_min:y_max, x_min:x_max]
        elif self._place == Place.GpuMat:
            data = cv2.cuda.GpuMat(self.data, rect.totuple())
        elif self._place == Place.UMat:
            data = cv2.UMat(self.data, rect.totuple())

        else:
            raise TypeError("Unknown place:'{}', image_data={}, image_data_type".format(self._place, self.data, type(self.data)))

        return self._clone_with_params(data, clone=False)

    def threshold(self, code=cv2.THRESH_OTSU):
        """
        图片二值化

        Args:
            code: type of the threshold operation

        Returns:
             Image: 二值化后的图片
        """
        if self._place == Place.Mat:
            _, data = cv2.threshold(self.data, 0, 255, code)  # return: np.ndarray
            data = self._create_mat(data=data, shape=data.shape)

        elif self._place in (Place.Ndarray, Place.UMat):
            _, data = cv2.threshold(self.data, 0, 255, code)

        elif self._place == Place.GpuMat:
            if code in (cv2.THRESH_OTSU, cv2.THRESH_TRIANGLE):
                # cuda threshold不支持这两种方法,需要转换
                _, data = cv2.threshold(self.data.download(), 0, 255, code)
                mat = cv2.cuda.GpuMat(data.shape[::-1])
                mat.upload(data)
                data = mat
            else:
                _, data = cv2.cuda.threshold(self.data, 0, 255, code)
        else:
            raise TypeError("Unknown place:'{}', image_data={}, image_data_type".format(self._place, self.data, type(self.data)))

        return self._clone_with_params(data, clone=False)

    def rectangle(self, rect: Rect, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 1, lineType=cv2.LINE_8):
        """
        在图像上画出矩形, 注!绘制会在原图上进行,不会产生新的图片对象

        Args:
            rect(Rect): 需要截图的范围
            color(tuple): 表示矩形边框的颜色
            thickness(int): 形边框的厚度
            lineType(int): 线的类型

        Returns:
             None
        """
        pt1 = rect.tl
        pt2 = rect.br

        if self._place in (Place.Mat, Place.Ndarray, Place.UMat):
            cv2.rectangle(self.data, (pt1.x, pt1.y), (pt2.x, pt2.y), color=color, thickness=thickness, lineType=lineType)
        elif self._place == Place.GpuMat:
            data = cv2.rectangle(self.data.download(), (pt1.x, pt1.y), (pt2.x, pt2.y), color=color, thickness=thickness, lineType=lineType)
            self.data.upload(data)
        else:
            raise TypeError("Unknown place:'{}', image_data={}, image_data_type".format(self._place, self.data, type(self.data)))

    def gaussianBlur(self, size: Tuple[int, int], sigma: Union[int, float]):
        """
        使用高斯滤镜模糊图像

        Args:
            size(tuple): 高斯核大小
            sigma(int|float): 高斯核标准差
        Returns:
             Image: 高斯滤镜模糊图像
        """
        # r, c = np.mgrid[0:size:1, 0:size:1]
        # r -= int((size - 1) / 2)
        # c -= int((size - 1) / 2)
        # norm_ = np.power(r, 2.0) + np.power(c, 2.0)
        # gaussianKernel = np.exp(- norm_ / (2 * sigma))
        # gaussianKernel /= np.sum(gaussianKernel)
        #
        # out = cv2.filter2D(img, cv2.CV_32F, kernel=gaussianKernel)
        # out = np.array(out, dtype=np.uint8)
        if self._place in (Place.Mat, Place.Ndarray, Place.UMat):
            return cv2.GaussianBlur(self.data, ksize=size, sigmaX=sigma)
        elif self._place == Place.GpuMat:
            pass
            # cv2.cuda.createGaussianFilter(self.data.download(), 0, 255)
        else:
            raise TypeError("Unknown place:'{}', image_data={}, image_data_type".format(self._place, self.data, type(self.data)))

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

        if self._place == Place.Mat:
            cv2.imshow(title, self.data)
        elif self._place in (Place.Mat, Place.Ndarray, Place.UMat):
            cv2.imshow(title, self.data)
        elif self._place == Place.GpuMat:
            cv2.imshow(title, self.data.download())

    # def imread(img_path) -> paddle.Tensor:
    #     img = Image(img_path, flags=cv2.IMREAD_UNCHANGED).imread()
    #     return paddle.to_tensor(img.transpose(2, 0, 1)[None, ...], dtype=paddle.float32)
    #
    #
    # img1 = imread('./image/0.png')
    # img2 = imread('./image/0.png')
    #
    # ssim(img1, img2, data_range=255) v
