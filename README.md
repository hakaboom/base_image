# base_image
对opencv_python常用接口的二次开发

兼容了cuda与numpy两种格式的图像数据

## Example

1. **Create**

Create an object from test.png

```Python
import cv2
from baseImage import Image

Image(img='test.png', flags=cv2.IMREAD_COLOR, path='./')
# param img: can be fileName/bytes/numpy.ndarry/cuda_GpuMat
# param flags: 'https://docs.opencv.org/master/d8/d6a/group__imgcodecs__flags.html#ga61d9b0126a3e57d9277ac48327799c80'
# param path: Used to set the read path
```

2. **transform_gpu**

Transform image Data to cuda_GpuMat

```Python
from baseImage import Image

img = Image('test.png')
img.transform_gpu()
```

3. **transform_cpu**

Transform image Data to numpy.ndarray

```Python
from baseImage import Image

img = Image('test.png')
img.transform_cpu()
```

4. **imread**

This returns image Data with numpy.ndarry

This function will call transform_cpu

```Python
from baseImage import Image

img = Image('test.png')
img.imread()
```

5. **download**

This returns image Data with cuda_GpuMat

This function will call transform_gpu

```Python
from baseImage import Image

img = Image('test.png')
img.download()
```

6. **imwrite**

Write Data to object

```Python
import cv2
from baseImage import Image

img = Image(path='./')
img.imwrite(data='test.png', flags=cv2.IMREAD_COLOR)

# param img: can be fileName/bytes/numpy.ndarry/cuda_GpuMat
# param flags: 'https://docs.opencv.org/master/d8/d6a/group__imgcodecs__flags.html#ga61d9b0126a3e57d9277ac48327799c80'
# param path: Used to set the read path
```

7. **shape**

This returns image shape with tuple

```Python
from baseImage import Image

img = Image('test.png')
print(img.shape)
# Output example：(1080, 1920, 4)
```

8. **size**

This return image size with tuple

```Python
from baseImage import Image

img = Image('test.png')
print(img.size)
# Output example：(1080, 1920)
```

9. **clone**

Returns a new clone object

```Python
from baseImage import Image

img1 = Image('test.png')
print(img1)
img2 = img1.clone()
print(img2)
print(img1 == img2)
# Output example：False
```

## More
[直接查看函数实现](https://github.com/hakaboom/base_image/blob/master/baseImage/base_image.py#L170)

