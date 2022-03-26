# -*- coding: utf-8 -*-
from .base_image import Image
from .coordinate import Rect, Point, Size
from .utils.ssim import SSIM
from .utils.image_diff import ImageDiff

name = 'base_image'


__all__ = ['Rect', 'Point', 'Size', 'Image', 'SSIM', 'ImageDiff']
