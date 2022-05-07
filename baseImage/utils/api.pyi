import numpy as np
from typing import Tuple, Union, overload, Type, List, Optional


Dtype = Union[Type[np.uint8], Type[np.int8], Type[np.uint16], Type[np.int16], Type[np.int32], Type[np.float32], Type[np.float64]]


def check_file(file_name: str) -> bool: ...


def check_image_valid(image: np.ndarray) -> bool: ...


def read_image(filename: str, flags: int) -> np.ndarray: ...


def bytes_2_img(byte: bytes) -> np.ndarray: ...


def npType_to_cvType(dtype: Dtype, channels: int) -> int: ...


def cvType_to_npType(dtype: int, channels: int) -> Dtype: ...


class AutoIncrement(object): ...