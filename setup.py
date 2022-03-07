# -*- coding: utf-8 -*-
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def install_requires():
    try:
        import cv2
        return None
    except ModuleNotFoundError:
        return [
            'numpy>=1.21.4',
            'opencv-python>=4.5.4.60',
            'pydantic'
        ]


setup(
    name='baseImage',
    version='1.1.1',
    author='hakaboom',
    author_email='1534225986@qq.com',
    license='Apache License 2.0',
    description='This is a secondary package of OpenCV,for manage image data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/hakaboom/base_image',
    packages=['baseImage'],
    install_requires=install_requires(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8, <=3.9',
)