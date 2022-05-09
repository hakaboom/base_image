"""
python setup.py sdist
twine upload dist/*
"""
import time

import cv2
from baseImage import Setting

cv2.cuda.setBufferPoolUsage(True)
cv2.cuda.setBufferPoolConfig(cv2.cuda.getDevice(), 1024 * 1024 * (3 + 3), 1)

stream = cv2.cuda.Stream()
pool = cv2.cuda.BufferPool(stream)

Setting.Default_Stream = stream
Setting.Default_Pool = pool
