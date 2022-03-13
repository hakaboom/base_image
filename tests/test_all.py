# -*- coding: utf-8 -*-
import unittest
from baseImage.base_image import Image, Size, Rect
from baseImage.constant import Place
import os

import numpy as np
import cv2

if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    cuda_flag = True
else:
    cuda_flag = False

try:
    cv2.cuda.GpuMat()
except AttributeError:
    cv2.cuda.GpuMat = cv2.cuda_GpuMat

IMAGEDIR = os.path.dirname(os.path.abspath(__file__)) + "/image"


class TestImage(unittest.TestCase):
    def setUp(self):
        if cuda_flag:
            self.place_list = [Place.Ndarray, Place.Mat,  Place.GpuMat, Place.UMat]
            self.place_type = [np.ndarray, cv2.Mat,  cv2.cuda.GpuMat, cv2.UMat]
        else:
            self.place_list = [Place.Ndarray, Place.Mat, Place.UMat]
            self.place_type = [np.ndarray, cv2.Mat, cv2.UMat]

    def test_create(self):
        for place in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)

            self.assertIsNotNone(img)
            self.assertIsNotNone(img.data)
            self.assertIsInstance(img.data, self.place_type[place])

    def test_image_clone(self):
        for place in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            new_img = img.clone()

            self.assertIsNotNone(new_img)
            self.assertIsNotNone(new_img.data)
            self.assertEqual(img.size, new_img.size)
            self.assertEqual(img.shape, new_img.shape)
            self.assertEqual(img.channels, new_img.channels)
            self.assertEqual(img.dtype, new_img.dtype)
            self.assertEqual(type(img.data), type(new_img.data))
            self.assertNotEqual(id(img.data), id(new_img.data))

    def test_resize(self):
        for place in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            new_img = img.resize(400, 400)
            self.assertEqual(new_img.size, (400, 400))
            self.assertIsInstance(new_img.data, self.place_type[place])

    def test_cvtColor(self):
        for place in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            new_img = img.cvtColor(cv2.COLOR_BGR2RGB)
            self.assertNotEqual(id(img.data), id(new_img.data))
            self.assertIsInstance(new_img.data, self.place_type[place])

    def test_crop(self):
        for place in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            rect = Rect(100, 100, 200, 200)
            new_img = img.crop(rect)
            self.assertEqual(new_img.size, (rect.size.width, rect.size.height))
            self.assertNotEqual(id(img.data), id(new_img.data))
            self.assertIsInstance(new_img.data, self.place_type[place])

    def test_threshold(self):
        for place in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            new_img = img.cvtColor(cv2.COLOR_BGR2GRAY).threshold()
            self.assertIsInstance(new_img.data, self.place_type[place])
            
    def test_rectangle(self):
        for place in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            img.rectangle(rect=Rect(100, 100, 200, 200))
            img.rectangle(rect=Rect(100, 100, 200, 200), color=(100, 100, 100), thickness=-2)


if __name__ == '__main__':
    unittest.main()
