from base_image import IMAGE
import cv2
from coordinate import Rect

a = IMAGE(img='test.png')
a.imshow()
cv2.waitKey(0)

b = a.crop_image(rect=Rect(0, 0, 100, 100))
b.imshow()
cv2.waitKey(0)

c = a.binarization()
c.imshow()
cv2.waitKey(0)

d = IMAGE(a)
d.crop_image(rect=Rect(0, 0, 200, 200))
print(d.shape)
print(b.shape)