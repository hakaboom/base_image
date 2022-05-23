# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def install_requires():
    try:
        import cv2
        return None
    except ModuleNotFoundError:
        return [
            'numpy',
            'opencv-python>=4.5.5.64',
            'pydantic'
        ]


setup(
    name='baseImage',
    version='2.1.2',
    author='hakaboom',
    author_email='1534225986@qq.com',
    license='Apache License 2.0',
    description='This is a secondary package of OpenCV,for manage image data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/hakaboom/base_image',
    include_package_data=True,
    packages=find_packages(),
    install_requires=install_requires(),
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6, <=3.10',
)