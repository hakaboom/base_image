"""
python setup.py sdist
twine upload dist/*
"""
import cv2
import time
import numpy as np
import numpy.typing as npt
from baseImage import Image, Rect, Place, Setting, SSIM, ImageDiff
import matplotlib.pyplot as plt
from baseImage.utils.image_diff import ImageSimilarity

test1 = Image('tests/image/test4.png')#.cvtColor(cv2.COLOR_BGR2HSV)
test2 = Image('tests/image/test5.png')#.cvtColor(cv2.COLOR_BGR2HSV)

hist1 = test1.calcHist([256], [0, 256])
hist2 = test2.calcHist([256], [0, 256])
a = ImageSimilarity()
print(a.diff(test1, test2))
# hist1 = [cv2.normalize(data, data, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) for data in hist1]
# hist2 = [cv2.normalize(data, data, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) for data in hist2]
# methods = [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA]
# for method in methods:
#     comparison_b = cv2.compareHist(hist1[0], hist2[0], method)
#     comparison_g = cv2.compareHist(hist1[1], hist2[1], method)
#     comparison_r = cv2.compareHist(hist1[2], hist2[2], method)
#     # print(f"methods={method}, b={comparison_b}")
#     print(f"methods={method}, b={comparison_b}, g={comparison_g}, r={comparison_r}")


# img = cv2.imread('tests/image/test4.png')    # Load the image
# channels = cv2.split(img)       # Set the image channels
# colors = ("b", "g", "r")        # Initialize tuple
# plt.figure()
# plt.title("Color Histogram")
# plt.xlabel("Bins")
# plt.ylabel("Number of Pixels")
#
# for (i, col) in zip(channels, colors):       # Loop over the image channels
#     hist = cv2.calcHist([i], [0], None, [256], [0, 256])   # Create a histogram for current channel
#     plt.plot(hist, color=col)      # Plot the histogram
#     plt.xlim([0, 256])
#
#
# color = ('blue', 'green', 'red')
#
# for i, color in enumerate(color):
#     hist = cv2.calcHist([test2.data], [i], None, [256], [0, 255])
#     hist = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
#     plt.plot(hist, color=color)
#     plt.xlim([0, 256])
#
# plt.show()
# cv2.waitKey()