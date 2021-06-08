#! usr/bin/python
# -*- coding:utf-8 -*-
"""
坐标系转换---从原来叉叉助手框架转移过来的
包含了锚点模式,适用于各种分辨率,刘海屏的坐标适配
"""
from typing import Union
from loguru import logger


class Point(object):
    """
        Point.ZERO      :一个x,y均为0的Point
        Point.INVALID   :一个x,y均为-1的Point
        Point(void) :构造一个x,y均为0的Point
        Point(x:int , y:int)    :根据x,y构造一个Point
        Point(Point)    :根据point,拷贝一个新的Point
        Point.x :x坐标
        Point.y :y坐标
        支持 +,-,*,/,==操作
    """
    def __init__(self, x: int, y: int,
                 anchor_mode: str = 'Middle', anchor_x: int = 0, anchor_y: int = 0):
        """
        构建一个点
        :param x: x轴坐标
        :param y: y轴坐标
        :param kwargs:
        """
        self.x = x
        self.y = y
        self.anchor_mode = anchor_mode
        self.anchor_x = anchor_x
        self.anchor_y = anchor_y

    def __str__(self):
        return '<Point [{:.1f}, {:.1f}]>'.format(self.x, self.y)

    def __add__(self, other):
        if type(other) == Point:
            return Point(self.x + other.x, self.y + other.y)
        raise logger.error('目标对象不是Point类,请检查')

    def __sub__(self, other):
        if type(other) == Point:
            return Point(self.x - other.x, self.y - other.y)
        raise logger.error('目标对象不是Point类,请检查')

    def __mul__(self, other):
        if type(other) == int:
            return Point(self.x * other, self.y * other)
        raise logger.error('目标对象不是int类,请检查')

    def __truediv__(self, other):
        if type(other) == int:
            return Point(self.x / other, self.y / other)
        raise logger.error('目标对象不是int类,请检查')

    def __eq__(self, other):
        if type(other) == Point:
            return self.x == other.x and self.y == other.y
        else:
            logger.error('目标对象不是Point类,请检查')
            return False


Point.ZERO = Point(0, 0)
Point.INVALID = Point(-1, -1)


class Size(object):
    """
        Size.ZERO      :一个width,height均为0的Size
        Size.INVALID   :一个width,height均为-1的Size
        Size(void) :构造一个width,height均为0的Size
        Size(width:int , height:int)    :根据width,height构造一个Size
        Size(Size)    :根据Size,拷贝一个新的Size
        Size.width  :Size的宽
        Size.height :Size的高
        支持 +,-,*,/,==操作
    """
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def __str__(self):
        return '<Size [{} x {}]>'.format(self.width,self.height)

    def __add__(self, other):
        if type(other) == Size:
            return Size(self.width + other.width, self.height + other.height)
        raise logger.error('目标对象不是Size类,请检查')

    def __sub__(self, other):
        if type(other) == Size:
            return Size(self.width - other.width, self.height - other.height)
        raise logger.error('目标对象不是Size类,请检查')

    def __mul__(self, other):
        if type(other) == int:
            return Size(self.width * other, self.height * other)
        raise logger.error('目标对象不是int类,请检查')

    def __truediv__(self, other):
        if type(other) == int:
            return Size(self.width / other, self.height / other)
        raise logger.error('目标对象不是int类,请检查')

    def __eq__(self, other):
        if type(other) == Point:
            return self.width == other.width and self.height == other.height
        else:
            logger.error('目标对象不是Size类,请检查')
            return False

    def __lt__(self, other):
        if type(other) == Size:
            return self.width*self.height < other.width*other.height
        else:
            logger.error('目标对象不是Size类,请检查')
            return False

    def __gt__(self, other):
        if type(other) == Size:
            return self.width*self.height > other.width*other.height
        else:
            logger.error('目标对象不是Size类,请检查')
            return False

    def __le__(self, other):
        if type(other) == Size:
            return self.width*self.height <= other.width*other.height
        else:
            logger.error('目标对象不是Size类,请检查')
            return False

    def __ge__(self, other):
        if type(other) == Size:
            return self.width*self.height >= other.width*other.height
        else:
            logger.error('目标对象不是Size类,请检查')
            return False


Size.ZERO = Size(0, 0)
Size.INVALID = Size(-1, -1)


class Rect(object):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __str__(self):
        return '<Rect [Point({}, {}), Size[{}, {}]]'.format(
            self.x, self.y, self.width, self.height)

    @property
    def size(self):
        return Size(self.width, self.height)

    @property
    def tl(self):
        """返回当前Rect的左上角Point坐标"""
        return Point(self.x, self.y)

    @property
    def br(self):
        """返回当前Rect的右下角Point坐标"""
        return Point(self.x+self.width, self.y+self.height)

    @property
    def middle(self):
        return Point(self.x+self.width/2, self.y+self.height/2)

    def contains(self, v):
        """判断Point,或者Rect是否在当前Rect范围中"""
        if isinstance(v, Point):
            tl, br = self.tl, self.br
            if tl.x <= v.x <= br.x and tl.y <= v.y <= br.y:
                return True
        elif isinstance(v, Rect):
            """判断左上,右下顶点坐标即可"""
            if self.contains(v.tl) and self.contains(v.br):
                return True
        return False

    @staticmethod
    def create_by_point_size(point: Point, size: Size):
        return Rect(point.x, point.y, size.width, size.height)

    @staticmethod
    def create_by_2_point(tl_point: Point, br_point: Point):
        return Rect(tl_point.x, tl_point.y, br_point.x-tl_point.x, br_point.y-tl_point.y)


Rect.ZERO = Rect(0, 0, 0, 0)


class Anchor_transform(object):
    @staticmethod
    def Middle(x, y, dev, cur, mainPoint_scale):
        x = cur.x / 2 - ((dev.x / 2 - x) * mainPoint_scale['x']) + cur.left
        y = cur.y / 2 - ((dev.y / 2 - y) * mainPoint_scale['y']) + cur.top
        return x, y

    @staticmethod
    def Left(x, y, dev, cur, mainPoint_scale):
        x = x * mainPoint_scale['x'] + cur.left
        y = cur.y/2-((dev.y/2-y)*mainPoint_scale['y'])+cur.top
        return x, y

    @staticmethod
    def Right(x, y, dev, cur, mainPoint_scale):
        x = cur.x-((dev.x-x) * mainPoint_scale['x'])+cur.left
        y = cur.y/2-((dev.y/2-y) * mainPoint_scale['y'])+cur.top
        return x, y

    @staticmethod
    def top(x, y, dev, cur, mainPoint_scale):
        x = cur.x / 2 - ((dev.x / 2 - x) * mainPoint_scale['x']) + cur.left
        y = y * mainPoint_scale['y'] + cur.top
        return x, y

    @staticmethod
    def Bottom(x, y, dev, cur, mainPoint_scale):
        x = cur.x / 2 - ((dev.x / 2 - x) * mainPoint_scale['x']) + cur.left
        y = cur.y - ((dev.y - y) * mainPoint_scale['y']) + cur.top
        return x, y

    @staticmethod
    def Left_top(x, y, dev, cur, mainPoint_scale):
        x = x * mainPoint_scale['x'] + cur.left
        y = y * mainPoint_scale['y'] + cur.top
        return x, y

    @staticmethod
    def Left_bottom(x, y, dev, cur, mainPoint_scale):
        x = x * mainPoint_scale['x'] + cur.left
        y = cur.y - ((dev.y - y) * mainPoint_scale['y']) + cur.top
        return x, y

    @staticmethod
    def Right_top(x, y, dev, cur, mainPoint_scale):
        x = cur.x - ((dev.x - x) * mainPoint_scale['x']) + cur.left
        y = y * mainPoint_scale['y'] + cur.top
        return x, y

    @staticmethod
    def Right_bottom(x, y, dev, cur, mainPoint_scale):
        """锚点右下"""
        x = cur.x - ((dev.x-x)*mainPoint_scale['x']) + cur.left
        y = cur.y - ((dev.y-y)*mainPoint_scale['y']) + cur.top
        return x, y
