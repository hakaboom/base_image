# -*- coding: utf-8 -*-
from setuptools import setup

setup(
    name='baseImage',
    version='1.0.5',
    author='hakaboom',
    author_email='1534225986@qq.com',
    license='Apache License 2.0',
    description='This is a secondary package of OpenCV,for manage image data',
    url='https://github.com/hakaboom/base_image',
    packages=['baseImage'],
    install_requires=['colorama>=0.4.4',
                      "loguru>=0.5.3",
                      "pydantic",
],
)