"""
python setup.py sdist
twine upload dist/*
"""

import cv2
from baseImage import Image

Image('./test.png').binarization().imshow()

img_path = './test.png'
pkgs = [
    ('cpu', cv2.gapi.core.cpu.kernels()),
]
in_mat = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2GRAY)

g_in = cv2.GMat()
g_sc = cv2.GScalar()
mat, threshold = cv2.gapi.threshold(g_in, g_sc, cv2.THRESH_OTSU)
comp = cv2.GComputation(cv2.GIn(g_in, g_sc), cv2.GOut(mat, threshold))
for pkg_name, pkg in pkgs:
    actual_mat, actual_thresh = comp.apply(cv2.gin(in_mat, (255, 255)), args=cv2.gapi.compile_args(pkg))
    Image(actual_mat).imshow()
    cv2.waitKey(0)