# -*- coding: utf-8 -*-
import unittest
from baseImage.base_image import Image, Size, Rect
from baseImage.constant import Place

import cv2

if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    cuda_flag = True
else:
    cuda_flag = False


class TestImage(unittest.TestCase):
    def setUp(self):
        if cuda_flag:
            self.place_list = [Place.Ndarray, Place.Mat, Place.UMat, Place.GpuMat]
        else:
            self.place_list = [Place.Ndarray, Place.Mat, Place.UMat]

    def clone_test(self, old_img, new_img):
        self.assertIsNotNone(new_img)
        self.assertIsNotNone(new_img.data)
        self.assertEqual(old_img.size, new_img.size)
        self.assertEqual(old_img.shape, new_img.shape)
        self.assertEqual(old_img.channels, new_img.channels)
        self.assertEqual(old_img.dtype, new_img.dtype)
        self.assertEqual(type(old_img.data), type(new_img.data))
        self.assertNotEqual(id(old_img.data), id(new_img.data))

    def test_create(self):
        for place in self.place_list:
            img = Image(data='./image/0.png', place=place)
            self.assertIsNotNone(img)
            self.assertIsNotNone(img.data)

    def test_image_clone(self):
        for place in self.place_list:
            img = Image(data='./image/0.png', place=place)
            new_img = img.clone()
            self.clone_test(old_img=img, new_img=new_img)

    def test_resize(self):
        for place in self.place_list:
            img = Image(data='./image/0.png', place=place)
            img = img.resize(400, 400)
            self.assertEqual(img.size, (400, 400))

        for place in self.place_list:
            img = Image(data='./image/0.png', place=place)
            img = img.resize(Size(400, 400))
            self.assertEqual(img.size, (400, 400))

    def test_cvtColor(self):
        for place in self.place_list:
            img = Image(data='./image/0.png', place=place)
            img = img.cvtColor(cv2.COLOR_BGR2RGB)

    def test_crop(self):
        for place in self.place_list:
            img = Image(data='./image/0.png', place=place)
            rect = Rect(100, 100, 200, 200)
            img = img.crop(rect)
            self.assertEqual(img.size, (rect.size.width, rect.size.height))

    def test_threshold(self):
        for place in self.place_list:
            img = Image(data='./image/0.png', place=place)
            img = img.cvtColor(cv2.COLOR_BGR2GRAY)
            img.threshold()
            
        
if __name__ == '__main__':
    unittest.main()
