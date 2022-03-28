# -*- coding: utf-8 -*-
import cv2
import numpy as np

from typing import Tuple, Union

from .constant import Place, SHOW_INDEX
from .coordinate import Rect, Size
from .utils.api import read_image, bytes_2_img, cvType_to_npType, npType_to_cvType


try:
    cv2.cuda.GpuMat()
except AttributeError:
    cv2.cuda.GpuMat = cv2.cuda_GpuMat


class _Image(object):
    def __init__(self, data, read_mode=cv2.IMREAD_COLOR, dtype=np.uint8, place=Place.Mat, clone=True):
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

    def write(self, data, read_mode=None, dtype=None, place=None, clone=True):
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

        if isinstance(data, _Image):
            data = data.data

        """
        这边逻辑我也觉得有点诡异
        1、当data是字符串或者字节流的时候,转换成np.ndarray
        2、当data是np.ndarray|cv2.Mat|cv2.cuda.GpuMat|cv2.UMat时
            if clone: 拷贝一份新的
            if not clone: 不拷贝
        3、根据place转换data的类型
        4、根据dtype转换data的数据类型
        """
        if isinstance(data, (str, bytes)):  # data: np.ndarray
            if isinstance(data, str):
                data = read_image(data, flags=read_mode)
            elif isinstance(data, bytes):
                data = bytes_2_img(data)
        else:
            if clone:
                if isinstance(data, np.ndarray):
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
    def _create_mat(cls, data, shape):
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
    def shape(self):
        """
        获取图片的长、宽、通道数

        Returns:
            shape: (长,宽,通道数)
        """
        if self.place in (Place.Mat, Place.Ndarray):
            shape = self.data.shape
        elif self.place == Place.GpuMat:
            shape = self.data.size()[::-1] + (self.data.channels(),)
        elif self.place == Place.UMat:
            shape = self.data.get().shape
        else:
            raise TypeError("Unknown place:'{}', image_data={}, image_data_type".format(self.place, self.data, type(self.data)))

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
        return self.shape[:-1]

    @property
    def channels(self):
        """
        获取图片的通道数

        Returns:
            channels: 通道数
        """
        if self.place in (Place.Mat, Place.Ndarray):
            return self.shape[2]
        elif self.place == Place.GpuMat:
            return self.data.channels()
        elif self.place == Place.UMat:
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
    def place(self):
        return self._place

    @property
    def data(self):
        return self._data


class Image(_Image):
    def clone(self):
        """
        拷贝一个新图片对象

        Returns:
            Image: 新图片对象
        """
        return Image(data=self._data, read_mode=self._read_mode, dtype=self.dtype, place=self.place)

    def _clone_with_params(self, data, **kwargs):
        """
        用data拷贝一个新图片对象(保留原Image对象的dtype|read_mode|place)

        Args:
            data: 图片数据

        Returns:
            Image: 新图片对象
        """
        clone = kwargs.pop('clone', True)
        return Image(data=data, read_mode=self._read_mode, dtype=self.dtype, place=self.place, clone=clone)

    def resize(self, *args, **kwargs):
        code = kwargs.get('code', cv2.INTER_LINEAR)

        if kwargs.get('size'):
            size = kwargs.get('size')
            if isinstance(size, Size):  # args: (size=Size(100, 100), code=2)
                w, h = size.width, size.height
            elif isinstance(size, (tuple, list)):  # args: (size=(100, 100), code=2)
                w, h = size
            else:
                raise ValueError('Unknown params args={}, kwargs={}'.format(args, kwargs))
        elif kwargs.get('w') and kwargs.get('h'):  # args: (w=100, h=100, code=2)
            w, h = kwargs.get('w'), kwargs.get('h')
        else:
            args_len = len(args)
            if args_len in (1, 2):
                if args_len == 1:  # args:(Size(100, 100)) or ((100, 100))
                    arg = args[0]
                else:
                    if isinstance(args[0], (Size, tuple, list)):  # args:(Size(100, 100),2) or ((100, 100), 2)
                        arg, code = args
                    else:  # args:(100, 100) or Size(100, 100)
                        arg = args

                if isinstance(arg, Size):
                    w, h = arg.width, arg.height
                elif isinstance(arg, (tuple, list)):
                    if len(arg) == 2:
                        w, h = arg
                    else:
                        raise ValueError('Unknown params args={}, kwargs={}'.format(args, kwargs))
                else:
                    raise ValueError('Unknown params args={}, kwargs={}'.format(args, kwargs))
            elif args_len == 3:
                w, h, code = args
            else:
                raise ValueError('Unknown params args={}, kwargs={}'.format(args, kwargs))

        assert type(w) == int, '参数必须是int类型, args={}, kwargs={}'.format(args, kwargs)
        assert type(h) == int, '参数必须是int类型 args={}, kwargs={}'.format(args, kwargs)
        assert type(code) == int, '参数必须是int类型 args={}, kwargs={}'.format(args, kwargs)

        return self._resize(w=w, h=h, code=code)

    def _resize(self, w, h, code=cv2.INTER_LINEAR):
        size = (w, h)
        if self.place == Place.Mat:
            data = cv2.resize(self.data, size, interpolation=code)  # return: np.ndarray
            data = self._create_mat(data, data.shape)
        elif self.place in (Place.Ndarray, Place.UMat):
            data = cv2.resize(self.data, size, interpolation=code)
        elif self.place == Place.GpuMat:
            data = cv2.cuda.resize(self.data, size, interpolation=code)
        else:
            raise TypeError("Unknown place:'{}', image_data={}, image_data_type".format(self.place, self.data, type(self.data)))
        return self._clone_with_params(data, clone=False)

    def cvtColor(self, code):
        """
        转换图片颜色空间

        Args:
            code(int): 颜色转换代码
                https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0

        Returns:
            Image: 转换后的新图片
        """
        if self.place == Place.Mat:
            data = cv2.cvtColor(self.data, code)  # return np.ndarray
            data = self._create_mat(data, shape=data.shape)
        elif self.place in (Place.Ndarray, Place.UMat):
            data = cv2.cvtColor(self.data, code)
        elif self.place == Place.GpuMat:
            data = cv2.cuda.cvtColor(self.data, code)
        else:
            raise TypeError("Unknown place:'{}', image_data={}, image_data_type".format(self.place, self.data, type(self.data)))

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

        if self.place in (Place.Mat, Place.Ndarray):
            x_min, y_min = int(rect.tl.x), int(rect.tl.y)
            x_max, y_max = int(rect.br.x), int(rect.br.y)
            data = self.data.copy()[y_min:y_max, x_min:x_max]
        elif self.place == Place.GpuMat:
            data = cv2.cuda.GpuMat(self.data, rect.totuple())
        elif self.place == Place.UMat:
            data = cv2.UMat(self.data, rect.totuple())
        else:
            raise TypeError("Unknown place:'{}', image_data={}, image_data_type".format(self.place, self.data, type(self.data)))

        return self._clone_with_params(data, clone=False)

    def threshold(self, thresh=0, maxval=255, code=cv2.THRESH_OTSU):
        """
        图片二值化

        Args:
            thresh: 阈值
            maxval: 最大值
            code: type of the threshold operation

        Returns:
             Image: 二值化后的图片
        """
        if self.place == Place.Mat:
            _, data = cv2.threshold(self.data, thresh, maxval, code)  # return: np.ndarray
            data = self._create_mat(data=data, shape=data.shape)
        elif self.place in (Place.Ndarray, Place.UMat):
            _, data = cv2.threshold(self.data, thresh, maxval, code)
        elif self.place == Place.GpuMat:
            if code > 4:
                # cuda threshold不支持这两种方法,需要转换
                _, data = cv2.threshold(self.data.download(), thresh, maxval, code)
            else:
                _, data = cv2.cuda.threshold(self.data, thresh, maxval, code)
        else:
            raise TypeError("Unknown place:'{}', image_data={}, image_data_type".format(self.place, self.data, type(self.data)))
        return self._clone_with_params(data, clone=False)

    def rectangle(self, rect, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_8):
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

        if self.place in (Place.Mat, Place.Ndarray, Place.UMat):
            cv2.rectangle(self.data, (pt1.x, pt1.y), (pt2.x, pt2.y), color=color, thickness=thickness, lineType=lineType)
        elif self.place == Place.GpuMat:
            data = cv2.rectangle(self.data.download(), (pt1.x, pt1.y), (pt2.x, pt2.y), color=color, thickness=thickness, lineType=lineType)
            self.data.upload(data)
        else:
            raise TypeError("Unknown place:'{}', image_data={}, image_data_type".format(self.place, self.data, type(self.data)))

    def copyMakeBorder(self, top, bottom, left, right, borderType):
        """
        扩充边缘

        Args:
            top(int): 上扩充大小
            bottom(int): 下扩充大小
            left(int): 左扩充大小
            right(int): 右扩充大小
            borderType(int): 边缘扩充类型

        Returns:

        """
        if self.place in (Place.Mat, Place.Ndarray, Place.UMat):
            data = cv2.copyMakeBorder(self.data, top, bottom, left, right, borderType)
        elif self.place == Place.GpuMat:
            data = cv2.cuda.copyMakeBorder(self.data, top, bottom, left, right, borderType)
        else:
            raise TypeError("Unknown place:'{}', image_data={}, image_data_type".format(self.place, self.data, type(self.data)))
        return self._clone_with_params(data, clone=False)

    def gaussianBlur(self, size=(11, 11), sigma=1.5, borderType=cv2.BORDER_DEFAULT):
        """
        使用高斯滤镜模糊图像

        Args:
            size(tuple): 高斯核大小
            sigma(int|float): 高斯核标准差
            borderType(int): pixel extrapolation method,
        Returns:
             Image: 高斯滤镜模糊图像
        """
        if not (size[0] % 2 == 1) or not (size[1] % 2 == 1):
            raise ValueError('Window size must be odd.')

        if self.place == Place.Mat:
            data = cv2.GaussianBlur(self.data, ksize=size, sigmaX=sigma, sigmaY=sigma, borderType=borderType)
            data = self._create_mat(data=data, shape=data.shape)
        elif self.place in (Place.Ndarray, Place.UMat):
            data = cv2.GaussianBlur(self.data, ksize=size, sigmaX=sigma, sigmaY=sigma, borderType=borderType)
        elif self.place == Place.GpuMat:
            dtype = self.data.type()
            # TODO: 感觉可以优化
            gaussian = cv2.cuda.createGaussianFilter(dtype, dtype, ksize=size, sigma1=sigma, sigma2=sigma,
                                                     rowBorderMode=borderType, columnBorderMode=borderType)
            data = gaussian.apply(self.data)
        else:
            raise TypeError("Unknown place:'{}', image_data={}, image_data_type".format(self.place, self.data, type(self.data)))
        return self._clone_with_params(data, clone=False)

    def bitwise_not(self, mask=None):
        """
        反转图片颜色

        Args:
            mask: 掩码

        Returns:
             Image: 反转后的图片
        """
        if self.place in (Place.Mat, Place.Ndarray, Place.UMat):
            data = cv2.bitwise_not(self.data, mask=mask)
        elif self.place == Place.GpuMat:
            data = cv2.cuda.bitwise_not(self.data, mask=mask)
        else:
            raise TypeError("Unknown place:'{}', image_data={}, image_data_type".format(self.place, self.data, type(self.data)))

        return self._clone_with_params(data, clone=False)

    def imshow(self, title=None, flag=cv2.WINDOW_KEEPRATIO):
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

        data = self.data
        if self.dtype != np.uint8:
            data = Image(data=data, dtype=np.uint8).data

        if isinstance(data, (np.ndarray, cv2.UMat)):
            cv2.imshow(title, data)
        elif isinstance(data, cv2.cuda.GpuMat):
            cv2.imshow(title, data.download())

    def imwrite(self, file_name):
        """
        讲图片保存到指定路径

        Args:
            file_name: 文件路径
            write_mode: 写入模式
                https://docs.opencv.org/4.x/d8/d6a/group__imgcodecs__flags.html#ga292d81be8d76901bff7988d18d2b42ac

        Returns:
            None
        """
        data = self.data
        if isinstance(data, (np.ndarray, cv2.UMat)):
            cv2.imwrite(file_name, data)
        elif isinstance(data, cv2.cuda.GpuMat):
            cv2.imwrite(file_name, data.download())

    def split(self):
        """
        拆分图像通道

        Returns:
            拆分后的图像数据,不会对数据包装处理
        """
        if self.place in (Place.Mat, Place.Ndarray, Place.UMat):
            data = cv2.split(self.data)
        elif self.place == Place.GpuMat:
            data = cv2.cuda.split(self.data)
        else:
            raise TypeError("Unknown place:'{}', image_data={}, image_data_type".format(self.place, self.data, type(self.data)))
        return data
