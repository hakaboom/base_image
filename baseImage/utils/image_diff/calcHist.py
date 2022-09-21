from baseImage import Image
import cv2
import numpy as np


class ImageSimilarity(object):
    def __init__(self):
        """
        基于直方图的图片相似度对比
        """
        self.method = cv2.HISTCMP_CORREL

    def diff(self, im1: Image, im2: Image):
        hist1 = im1.calcHist([256], [0, 256])
        hist2 = im2.calcHist([256], [0, 256])

        hist1 = [cv2.normalize(data, data, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) for data in hist1]
        hist2 = [cv2.normalize(data, data, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) for data in hist2]

        bgr_confidence = []
        for i, color in enumerate(('blue', 'green', 'red')):
            val = cv2.compareHist(hist1[i], hist2[i], self.method)
            bgr_confidence.append(val)

        return np.mean(bgr_confidence)

# color = ('blue', 'green', 'red')
#
# for i, color in enumerate(color):
#     hist = cv2.calcHist([test2.data], [i], None, [256], [0, 255])
#     hist = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
#     plt.plot(hist, color=color)
#     plt.xlim([0, 256])