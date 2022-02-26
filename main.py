"""
python setup.py sdist
twine upload dist/*
"""

import cv2
from baseImage import Image, Rect
from baseImage.coordinate import Anchor, screen_display_type, scale_mode_type

anchor = Anchor(
    dev=screen_display_type(
        width=1920, height=1080),
    cur=screen_display_type(
        width=2532, height=1170, left=100, right=100
    ),
    orientation=1)


a = Rect.create_by_point_size(point=anchor.point(1562, 154, anchor_mode='Right'),
                                size=anchor.size(187, 208))

print((2 % 24) * 80 + 40 )