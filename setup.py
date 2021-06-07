# -*- coding: utf-8 -*-
from setuptools import setup
import setuptools

setup(
    name='baseImage',
    version='1.0.0',
    author='hakaboom',
    author_email='1534225986@qq.com',
    description='This is a secondary package of OpenCV,for manage image data',
    url='https://github.com/hakaboom/base_image',
    packages=setuptools.find_packages(),
    platform='any',
    install_requires=['colorama>=0.4.4',
                      "loguru>=0.5.3",
                      "numpy>=1.20.3",
                      "opencv-python>=4.5.2.54",
],
)