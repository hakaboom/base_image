# base_image
对opencv_python常用接口的二次开发

建议 opencv version >= 4.5.5(不同opencv版本的python绑定,函数名可能会不同)

# Example

## Create
1. 默认方式创建图片对象
```python
import cv2
from baseImage import Image
from baseImage.constant import Place
    
Image(data='tests/image/0.png')  # 使用默认方式创建
```

2. 通过其他参数,调整图片参数

- 使用place参数,修改数据格式
  - Ndarray: 格式为numpy.ndarray格式
  - Umat: python的绑定不多,没有ndarray灵活,可以用于opencl加速
  - GpuMat: opencv的cuda格式,需要注意显存消耗
    - 可以通过常量`Default_Pool`设定缓冲区

    ```python
    import cv2
    from baseImage import Setting
    
    cv2.cuda.setBufferPoolUsage(True)
    cv2.cuda.setBufferPoolConfig(cv2.cuda.getDevice(), 1024 * 1024 * (3 + 3), 1)
    
    stream = cv2.cuda.Stream()
    pool = cv2.cuda.BufferPool(stream)
    
    Setting.Default_Stream = stream
    Setting.Default_Pool = pool
    ```

```python
import cv2
from baseImage import Image
from baseImage.constant import Place
    
Image(data='tests/image/0.png', place=Place.Ndarray)  # 使用numpy
Image(data='tests/image/0.png', place=Place.UMat)  # 使用Umat
Image(data='tests/image/0.png', place=Place.GpuMat)  # 使用cuda
```

- 使用dtype,修改数据类型
```python
import cv2
import numpy as np
from baseImage.utils.api import cvType_to_npType, npType_to_cvType
from baseImage import Image
    
Image(data='tests/image/0.png', dtype=np.uint8)
Image(data='tests/image/0.png', dtype=np.int8)
Image(data='tests/image/0.png', dtype=np.uint16)
Image(data='tests/image/0.png', dtype=np.int16)
Image(data='tests/image/0.png', dtype=np.int32)
Image(data='tests/image/0.png', dtype=np.float32)
Image(data='tests/image/0.png', dtype=np.float64)
# cvType_to_npType和npType_to_cvType提供了numpy转opencv数据格式的方法, cv的数据格式意义自行百度
```

- clone,用于处理是否拷贝原数据
```python
import cv2
import numpy as np
from baseImage import Image, Rect

img1 = Image(data='tests/image/0.png')
img2 = Image(img1, clone=False)
img2.rectangle(rect=Rect(0, 0, 200, 200), color=(255, 0, 0), thickness=-1)
img2.imshow('img2')
img1.imshow('img1')
cv2.waitKey(0)
```

## property
1. shape: 获取图片的长、宽、通道数
```python
from baseImage import Image

img = Image(data='tests/image/0.png')
print(img.shape)
# expect output
#       (1037, 1920, 3)
```
2. size: 获取图片的长、宽
```python
from baseImage import Image

img = Image(data='tests/image/0.png')
print(img.size)
# expect output
#       (1037, 1920)
```
3. channels: 获取图片的通道数量
```python
from baseImage import Image

img = Image(data='tests/image/0.png')
print(img.channels)
# expect output
#       3
```
4. dtype: 获取图片的数据类型
```python
from baseImage import Image

img = Image(data='tests/image/0.png')
print(img.dtype)
# expect output
#       numpy.uint8
```
5. place: 获取图片的数据格式
```python
from baseImage import Image
from baseImage.constant import Place

img = Image(data='tests/image/0.png', place=Place.Ndarray)
print(img.place == Place.Ndarray)
# expect output
#       True
```
6. data: 获取图片数据
```python
from baseImage import Image

img = Image(data='tests/image/0.png')
print(img.data)
```

## Function
1. dtype_convert: 数据类型转换
  - 将修改原图像数据
```python
from baseImage import Image
import numpy as np

img = Image(data='tests/image/0.png', dtype=np.uint8)
print(img.dtype)
img.dtype_convert(dtype=np.float32)
print(img.dtype)
```
2. place_convert: 数据格式转换
  - 将修改原图像数据
```python
from baseImage import Image
from baseImage.constant import Place

img = Image(data='tests/image/0.png', place=Place.Ndarray)
print(img.place == Place.Ndarray)
img.place_convert(place=Place.UMat)
print(img.place == Place.Ndarray)
print(img.place == Place.UMat)
```

3. clone: 克隆一个新的图片对象
```python
from baseImage import Image
from baseImage.constant import Place

img = Image(data='tests/image/0.png', place=Place.Ndarray)
img2 = img.clone()
print(img == img2)
```

4. rotate: 旋转图片, 现在只支持opencv自带的三个方向
```python
from baseImage import Image
import cv2

img = Image(data='tests/image/0.png')
img.rotate(code=cv2.ROTATE_180).imshow('180')
img.rotate(code=cv2.ROTATE_90_CLOCKWISE).imshow('90_CLOCKWISE')
img.rotate(code=cv2.ROTATE_90_COUNTERCLOCKWISE).imshow('90_COUNTERCLOCKWISE')
cv2.waitKey(0)
```

5. resize: 缩放图像
```python
from baseImage import Image

img = Image(data='tests/image/0.png')
new_img = img.resize(200, 200)
print(new_img.size)
```

6. cvtColor: 转换图片颜色空间
```python
from baseImage import Image
import cv2

img = Image(data='tests/image/0.png')
new_img = img.cvtColor(cv2.COLOR_BGR2GRAY)
new_img.imshow()
cv2.waitKey(0)
```

7. crop: 裁剪图片
```python
from baseImage import Image, Rect
import cv2

img = Image(data='tests/image/0.png')
new_img = img.crop(rect=Rect(0, 0, 400, 400))
new_img.imshow()
cv2.waitKey(0)
```

8. threshold: 二值化图片
```python
from baseImage import Image
import cv2

img = Image(data='tests/image/0.png')
new_img = img.threshold(thresh=0, maxval=255, code=cv2.THRESH_OTSU)
new_img.imshow()
cv2.waitKey(0)
```

9. rectangle: 在图像上画出矩形
  - 会在原图上进行修改
```python
from baseImage import Image, Rect
import cv2

img = Image(data='tests/image/0.png')
img.rectangle(rect=Rect(100, 100, 300, 300), color=(255, 0, 0), thickness=-1)
img.imshow()
cv2.waitKey(0)
```

10. copyMakeBorder: 扩充图片边缘
```python
from baseImage import Image
import cv2

img = Image(data='tests/image/0.png')
new_img = img.copyMakeBorder(top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_REPLICATE)
new_img.imshow()
cv2.waitKey(0)
```

11. gaussianBlur: 高斯模糊
```python
from baseImage import Image
import cv2

img = Image(data='tests/image/0.png')
new_img = img.gaussianBlur(size=(11, 11), sigma=1.5, borderType=cv2.BORDER_DEFAULT)
new_img.imshow()
cv2.waitKey(0)
```

12. warpPerspective: 透视变换
```python
from baseImage import Image, Size
import cv2
import numpy as np

img = Image(data='tests/image/0.png')
point_1 = np.float32([[0, 0], [100, 0], [0, 200], [100, 200]])
point_2 = np.float32([[0, 0], [50, 0], [0, 100], [50, 100]])
matrix = cv2.getPerspectiveTransform(point_1, point_2)
size = Size(50, 100)

new_img = img.warpPerspective(matrix, size=size)
new_img.imshow()
cv2.waitKey(0)
```
13. bitwise_not: 反转图片颜色
```python
from baseImage import Image
import cv2

img = Image(data='tests/image/0.png')
new_img = img.bitwise_not()
new_img.imshow()
cv2.waitKey(0)
```

14. imshow: 以GUI显示图片
```python
from baseImage import Image
import cv2

img = Image(data='tests/image/0.png')
img.imshow('img1')
cv2.waitKey(0)
```

15. imwrite: 将图片保存到指定路径
```python
from baseImage import Image
import cv2

img = Image(data='tests/image/0.png').cvtColor(cv2.COLOR_BGR2GRAY)
img.imwrite('tests/image/0_gray.png')
```

16. split: 拆分图像通道
  - 会直接返回拆分后的数据,不是Image类型
```python
from baseImage import Image

img = Image(data='tests/image/0.png')
img_split = img.split()
```

## Extra
1. SSIM: 图片结构相似性
   - resize: 图片缩放大小 
```python
from baseImage import SSIM, Image

ssim = SSIM(resize=(600, 600))
img1 = Image('tests/image/0.png')
img2 = Image('tests/image/0.png')
print(ssim.ssim(im1=img1, im2=img2))
```

2. image_diff: 基于SSIM的图片差异对比
```python
from baseImage import ImageDiff, Image
import cv2

diff = ImageDiff()

img1 = Image('tests/image/0.png')
img2 = Image('tests/image/1.png') 
cnts = diff.diff(img1, img2)
imageA = img1.data.copy()
imageB = img2.data.copy()
print(len(cnts))
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.waitKey(0)

```