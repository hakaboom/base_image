# -*- coding: utf-8 -*-
import unittest
from baseImage import Image, Rect, Size
from baseImage.constant import Place, Setting
from baseImage.utils.api import cvType_to_npType
from baseImage.utils.ssim import SSIM
import os

import numpy as np
import cv2


IMAGEDIR = os.path.dirname(os.path.abspath(__file__)) + "/image"


class TestImage(unittest.TestCase):
    def setUp(self):
        if Setting.CUDA_Flag:
            self.place_list = (Place.Ndarray, Place.UMat, Place.GpuMat)
        else:
            self.place_list = (Place.Ndarray, Place.UMat)
        self.dtype_list = [np.uint8, np.int8, np.uint16, np.int16, np.int32, np.float32, np.float64]

    def assertDTypeEqual(self, image: Image, dtype):
        image_np_dtype = Image.get_np_dtype(image.data)

        self.assertEqual(image_np_dtype, dtype)
        self.assertEqual(image.dtype, dtype)

    def assertPlaceEqual(self, image: Image, place: Place):
        self.assertEqual(type(image.data), place)
        self.assertEqual(image.place, place)

    def assertImagePtrNotEqual(self, img: Image, new_img: Image):
        """
        判断两个图片的数据,内存指针不相等
        """
        self.assertEqual(img.place, new_img.place)
        if img.place == Place.Ndarray:
            self.assertIsNone(img.data.base)
            self.assertIsNone(new_img.data.base)
        elif img.place == Place.UMat:
            pass
        elif img.place == Place.GpuMat:
            self.assertNotEqual(img.data.cudaPtr(), new_img.data.cudaPtr())

    """
    test_case
    """
    def test_create(self):
        for place in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            self.assertPlaceEqual(img, place)  # 判断类型是否一致

    def test_dtype_convert(self):
        for place in self.place_list:
            for dtype in self.dtype_list:
                img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place, dtype=dtype)
                self.assertDTypeEqual(img, dtype)
                self.assertPlaceEqual(img, place)

    def test_place_convert(self):
        for place in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            for new_place in self.place_list:
                new_img = Image(data=img, place=new_place)

                self.assertPlaceEqual(new_img, new_place)

    def test_get_shape(self):
        for place in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)

            self.assertEqual(img.shape, (1037, 1920, 3))

    def test_get_size(self):
        for place in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)

            self.assertEqual(img.size, (1037, 1920))

    def test_get_dtype(self):
        for place in self.place_list:
            for dtype in self.dtype_list:
                img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place, dtype=dtype)

                self.assertEqual(img.dtype, dtype)

    def test_image_clone(self):
        for place in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            new_img = img.clone()

            self.assertPlaceEqual(img, place)
            self.assertPlaceEqual(new_img, place)

            self.assertEqual(img.size, new_img.size)
            self.assertEqual(img.shape, new_img.shape)
            self.assertEqual(img.channels, new_img.channels)
            self.assertEqual(img.dtype, new_img.dtype)
            self.assertImagePtrNotEqual(img, new_img)

    def test_resize(self):
        for place in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            new_img = img.resize(400, 400, cv2.INTER_AREA)

            self.assertPlaceEqual(new_img, place)

            self.assertEqual(new_img.size, (400, 400))
            self.assertImagePtrNotEqual(img, new_img)

    def test_cvtColor(self):
        codes = [(cv2.COLOR_BGR2RGB, 3), (cv2.COLOR_BGR2GRAY, 1), (cv2.COLOR_BGR2HSV, 3)]

        for place in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            for code, channels in codes:
                new_img = img.cvtColor(code)

                self.assertPlaceEqual(new_img, place)

                self.assertEqual(new_img.channels, channels)
                self.assertImagePtrNotEqual(img, new_img)

    def test_crop(self):
        rect = Rect(100, 100, 200, 200)
        for place in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            new_img = img.crop(rect)

            self.assertPlaceEqual(new_img, place)
            self.assertEqual(new_img.size, (rect.size.width, rect.size.height))
            self.assertImagePtrNotEqual(img, new_img)

    def test_threshold(self):
        for place in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            new_img = img.cvtColor(cv2.COLOR_BGR2GRAY).threshold()

            self.assertImagePtrNotEqual(img, new_img)
            self.assertPlaceEqual(new_img, place)

    def test_rectangle(self):
        for place in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            img.rectangle(rect=Rect(100, 100, 200, 200))
            img.rectangle(rect=Rect(500, 500, 100, 100), color=(255, 255, 255), thickness=0)

            self.assertPlaceEqual(img, place)

    def test_gaussianBlur(self):
        for place in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            new_img = img.gaussianBlur(size=(11, 11), sigma=1.5)

            self.assertImagePtrNotEqual(img, new_img)
            self.assertPlaceEqual(new_img, place)

    def test_rotate(self):
        for place in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            for code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
                new_img = img.rotate(code)

                self.assertImagePtrNotEqual(img, new_img)
                self.assertPlaceEqual(new_img, place)

    def test_split(self):
        for place in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            img.split()

    def test_copyMakeBorder(self):
        for place in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            new_img = img.copyMakeBorder(10, 10, 10, 10, cv2.BORDER_REPLICATE)

            size = Size(img.size[0] + 20, img.size[1] + 20)
            self.assertEqual(new_img.size, (size.width, size.height))
            self.assertImagePtrNotEqual(img, new_img)
            self.assertPlaceEqual(new_img, place)

    def test_warpPerspective(self):
        point_1 = np.float32([[0, 0], [100, 0], [0, 200], [100, 200]])
        point_2 = np.float32([[0, 0], [50, 0], [0, 100], [50, 100]])
        matrix = cv2.getPerspectiveTransform(point_1, point_2)
        size = Size(50, 100)
        for place in self.place_list:
            img = Image(data=os.path.join(IMAGEDIR, '0.png'), place=place)
            new_img = img.warpPerspective(matrix, size=size)

            self.assertImagePtrNotEqual(img, new_img)
            self.assertPlaceEqual(new_img, place)

    def test_ssim(self):
        for place in self.place_list:
            im1 = Image(data=os.path.join(IMAGEDIR, '1.png'), dtype=np.float32, place=place)
            im2 = Image(data=os.path.join(IMAGEDIR, '1.png'), dtype=np.float32, place=place)

            ssim = SSIM()
            m, S = ssim.ssim(im1, im2, full=True)
            self.assertIsNotNone(m)


if __name__ == '__main__':
    unittest.main()
