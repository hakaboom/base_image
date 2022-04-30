# -*- coding: utf-8 -*-
import cv2
import numpy as np

from typing import Tuple, Union, overload, Type, List, Optional

from .constant import Place
from .coordinate import Rect, Size


Dtype = Union[Type[np.uint8], Type[np.int8], Type[np.uint16], Type[np.int16], Type[np.int32], Type[np.float32], Type[np.float64]]
Stream = Optional[cv2.cuda.Stream]


class _Image(object):
    _data: Union[np.ndarray, cv2.cuda.GpuMat, cv2.Mat, cv2.UMat]
    _read_mode: int
    _dtype: Dtype
    _place: int
    def __init__(self, data: Union[str, bytes, np.ndarray, cv2.cuda.GpuMat, cv2.Mat, cv2.UMat, Image],
                 read_mode: int = cv2.IMREAD_COLOR,
                 dtype: Dtype = np.uint8,
                 place: int = Place.Ndarray, clone: bool = True): ...

    def write(self, data: Union[str, bytes, np.ndarray, cv2.cuda.GpuMat, cv2.Mat, cv2.UMat, Image],
              read_mode: int = None, dtype: Dtype = None, place=None, clone=True) -> None: ...

    @classmethod
    def _create_mat(cls, data: Union[np.ndarray, cv2.Mat], shape: Union[tuple, list]) -> cv2.Mat: ...

    def dtype_convert(self, dtype: Dtype) -> None: ...

    def place_convert(self, place: int) -> None: ...

    @property
    def shape(self) -> Tuple[int, int, int]: ...

    @property
    def size(self) -> Tuple[int, int]: ...

    @property
    def channels(self) -> int: ...

    @property
    def dtype(self): ...

    @property
    def place(self) -> int: ...

    @property
    def data(self) -> Union[np.ndarray, cv2.cuda.GpuMat, cv2.Mat, cv2.UMat]: ...


class Image(_Image):
    def clone(self) -> Image:
        pass

    def _clone_with_params(self, data, **kwargs) -> Image: ...

    @overload
    def resize(self, w: int, h: int, code: int = cv2.INTER_LINEAR, stream: Stream = None) -> Image: ...

    @overload
    def resize(self, size: Union[Tuple[int, int], List[int, int], Size], code: int = cv2.INTER_LINEAR, stream=Stream) -> Image: ...

    def rotate(self, code: int, stream: Stream = None) -> Image: ...

    def _resize(self, w: int, h: int, code: int = cv2.INTER_LINEAR, stream: Stream = None) -> Image: ...

    def cvtColor(self, code: int, stream: Stream = None) -> Image: ...

    def crop(self, rect: Rect) -> Image: ...

    def threshold(self, thresh: int = 0, maxval: int = 255, code=cv2.THRESH_OTSU, stream: Stream = None) -> Image: ...

    def rectangle(self, rect: Rect, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 1, lineType=cv2.LINE_8) -> None: ...

    def copyMakeBorder(self, top: int, bottom: int, left: int, right: int, borderType: int, stream: Stream = None) -> Image: ...

    def gaussianBlur(self, size: Tuple[int, int] = (0, 0), sigma: Union[int, float] = 1.5, borderType: int = cv2.BORDER_DEFAULT, stream: Stream = None) -> Image: ...

    def warpPerspective(self, matrix: np.ndarray, size: Union[Tuple[int, int], List[int, int], Size], flags: int = cv2.INTER_LINEAR, borderMode: int = cv2.BORDER_CONSTANT, borderValue: int = 0, stream: Stream = None) -> Image: ...

    def bitwise_not(self, mask=None, stream: Stream = None) -> Image: ...

    def imshow(self, title: str = None, flags: int = cv2.WINDOW_KEEPRATIO) -> None: ...

    def imwrite(self, fileName: str) -> None: ...

    def split(self, stream: Stream = None) -> Tuple[Union[np.ndarray, cv2.cuda.GpuMat, cv2.UMat], ...]: ...
