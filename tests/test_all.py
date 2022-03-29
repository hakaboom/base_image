# -*- coding: utf-8 -*-
import unittest
from baseImage import Image, Rect, Size
from baseImage.constant import Place, CUDA_Flag
from baseImage.utils.api import cvType_to_npType
from baseImage.utils.ssim import SSIM
import os

import numpy as np
import cv2


try:
    cv2.cuda.GpuMat()
except AttributeError:
    cv2.cuda.GpuMat = cv2.cuda_GpuMat

IMAGEDIR = os.path.dirname(os.path.abspath(__file__)) + "/image"


class TestImage(unittest.TestCase):
    def setUp(self):
        if CUDA_Flag:
            self.place_list = [(Place.Ndarray, np.ndarray), (Place.Mat, cv2.Mat), (Place.UMat, cv2.UMat), 
                               (Place.GpuMat, cv2.cuda_GpuMat)]
        else:
            self.place_list = [(Place.Ndarray, np.ndarray), (Place.Mat, cv2.Mat), (Place.UMat, cv2.UMat)]
        self.dtype_list = [np.uint8, np.int8, np.uint16, np.int16, np.int32, np.float32, np.float64]

    def test_create(self):
        for place, ptype in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)

            self.assertIsNotNone(img)
            self.assertIsNotNone(img.data)
            self.assertEqual(type(img.data), ptype)  # 判断类型是否一致

    def test_dtype_convert(self):
        for place, ptype in self.place_list:
            for dtype in self.dtype_list:
                img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place, dtype=dtype)
                if isinstance(img.data, cv2.cuda_GpuMat):
                    self.assertEqual(cvType_to_npType(img.data.type(), channels=img.channels), dtype)
                elif isinstance(img.data, cv2.UMat):
                    self.assertEqual(img.data.get().dtype, dtype)
                elif isinstance(img.data, np.ndarray):
                    self.assertEqual(img.data.dtype, dtype)

    def test_place_convert(self):
        for place, _ in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            for new_place, ptype in self.place_list:
                new_img = Image(data=img, place=new_place)

                self.assertIsNotNone(new_img)
                self.assertIsNotNone(new_img.data)
                self.assertNotEqual(id(img.data), id(new_img.data))
                self.assertEqual(type(new_img.data), ptype)  # 判断类型是否一致

    def test_get_shape(self):
        for place, ptype in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)

            self.assertEqual(img.shape, (1037, 1920, 3))

    def test_get_size(self):
        for place, ptype in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)

            self.assertEqual(img.size, (1037, 1920))

    def test_get_dtype(self):
        for place, ptype in self.place_list:
            for dtype in self.dtype_list:
                img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place, dtype=dtype)
                self.assertEqual(img.dtype, dtype)

    def test_image_clone(self):
        for place, ptype in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            new_img = img.clone()

            self.assertEqual(img.size, new_img.size)
            self.assertEqual(img.shape, new_img.shape)
            self.assertEqual(img.channels, new_img.channels)
            self.assertEqual(img.dtype, new_img.dtype)
            self.assertEqual(type(img.data), type(new_img.data))
            self.assertEqual(type(img.data), ptype)  # 判断类型是否一致
            self.assertNotEqual(id(img.data), id(new_img.data))

    def test_resize(self):
        for place, ptype in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            new_img = img.resize(400, 400, cv2.INTER_AREA)

            self.assertEqual(new_img.size, (400, 400))
            self.assertNotEqual(id(img.data), id(new_img.data))
            self.assertEqual(type(img.data), ptype)  # 判断类型是否一致

    def test_cvtColor(self):
        codes = [(cv2.COLOR_BGR2RGB, 3), (cv2.COLOR_BGR2GRAY, 1), (cv2.COLOR_BGR2HSV, 3)]

        for place, ptype in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            for code, channels in codes:
                new_img = img.cvtColor(code)

                self.assertEqual(new_img.channels, channels)
                self.assertNotEqual(id(img.data), id(new_img.data))
                self.assertEqual(type(img.data), ptype)  # 判断类型是否一致

    def test_crop(self):
        for place, ptype in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            rect = Rect(100, 100, 200, 200)
            new_img = img.crop(rect)

            self.assertEqual(new_img.size, (rect.size.width, rect.size.height))
            self.assertNotEqual(id(img.data), id(new_img.data))
            self.assertEqual(type(img.data), ptype)  # 判断类型是否一致

    def test_threshold(self):
        for place, ptype in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            new_img = img.cvtColor(cv2.COLOR_BGR2GRAY).threshold()

            self.assertNotEqual(id(img.data), id(new_img.data))
            self.assertEqual(type(img.data), ptype)  # 判断类型是否一致
            
    def test_rectangle(self):
        for place, ptype in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            img.rectangle(rect=Rect(100, 100, 200, 200))
            img.rectangle(rect=Rect(500, 500, 100, 100), color=(255, 255, 255), thickness=0)

            self.assertEqual(type(img.data), ptype)  # 判断类型是否一致

    def test_gaussianBlur(self):
        for place, ptype in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            new_img = img.gaussianBlur(size=(11, 11), sigma=1.5)

            self.assertNotEqual(id(img.data), id(new_img.data))
            self.assertEqual(type(img.data), ptype)  # 判断类型是否一致

    def test_rotate(self):
        for place, ptype in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            for code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
                img.rotate(code)

    def test_split(self):
        for place, ptype in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            img_bgr = img.split()
            # 好像没啥东西可以验证

    def test_copyMakeBorder(self):
        for place, ptype in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            new_img = img.copyMakeBorder(10, 10, 10, 10, cv2.BORDER_REPLICATE)

            size = Size(img.size[0] + 20, img.size[1] + 20)
            self.assertEqual(new_img.size, (size.width, size.height))
            self.assertNotEqual(id(img.data), id(new_img.data))
            self.assertEqual(type(img.data), ptype)  # 判断类型是否一致

    def test_ssim(self):
        for place, ptype in self.place_list:
            im1 = Image(data=os.path.join(IMAGEDIR, '1.png'), dtype=np.float32, place=place)
            im2 = Image(data=os.path.join(IMAGEDIR, '2.png'), dtype=np.float32, place=place)

            ssim = SSIM()
            m, S = ssim.ssim(im1, im2, full=True)
            self.assertIsNotNone(m)


if __name__ == '__main__':
    unittest.main()
