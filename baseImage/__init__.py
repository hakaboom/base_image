# -*- coding: utf-8 -*-
from .base_image import IMAGE
from .coordinate import Rect, Point, Size
import cv2
name = 'base_image'


def create(img=None, flags=cv2.IMREAD_COLOR, path=''):
    return IMAGE(img, flags, path)


__all__ = ['create', 'Rect', 'Point', 'Size', 'IMAGE']