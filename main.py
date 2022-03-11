"""
python setup.py sdist
twine upload dist/*
"""
from baseImage.base_image import Image
from baseImage.constant import Place

place_list = [Place.Ndarray, Place.Mat, Place.UMat, Place.GpuMat]


img = Image(data='tests/image/0.png', place=1)
print(img.size == (1037, 1920))
