"""
python setup.py sdist
twine upload dist/*
"""
import cv2

from baseImage import Image
# path = r'C:\Users\Administrator.hzq\Desktop\test\RPReplay_Final1648388373.MP4'
# save_path = r'C:\Users\Administrator.hzq\Desktop\test'
#
# video = cv2.VideoCapture(path)
# index = 0
# frame_index = 1
# if video.isOpened():
#     rval, frame = video.read()
# else:
#     rval = False
#
# while rval:
#     rval, frame = video.read()
#
#     if index % 10 == 0:
#         Image(frame).imwrite(os.path.join(save_path, f'{int(frame_index)}.png'))
#         frame_index += 1
#     index += 1

a = Image('tests/image/0.png', place=2)
a.rotate(cv2).imshow()
cv2.waitKey(0)