# Author: Tao Hu <taohu620@gmail.com>

# -*- coding: utf-8 -*-
# File: rect.py


import numpy as np
import cv2

__all__ = [ 'draw_det_box',
           'draw_boxes']

class BoxBase(object):
    __slots__ = ['x1', 'y1', 'x2', 'y2']

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def copy(self):
        new = type(self)()
        for i in self.__slots__:
            setattr(new, i, getattr(self, i))
        return new

    def __str__(self):
        return '{}(x1={}, y1={}, x2={}, y2={})'.format(
            type(self).__name__, self.x1, self.y1, self.x2, self.y2)

    __repr__ = __str__

    def area(self):
        return self.w * self.h

    def is_box(self):
        return self.w > 0 and self.h > 0


class IntBox(BoxBase):
    def __init__(self, x1, y1, x2, y2):
        for k in [x1, y1, x2, y2]:
            assert isinstance(k, int)
        super(IntBox, self).__init__(x1, y1, x2, y2)

    @property
    def w(self):
        return self.x2 - self.x1 + 1

    @property
    def h(self):
        return self.y2 - self.y1 + 1

    def is_valid_box(self, shape):
        """
        Check that this rect is a valid bounding box within this shape.

        Args:
            shape: int [h, w] or None.
        Returns:
            bool
        """
        if min(self.x1, self.y1) < 0:
            return False
        if min(self.w, self.h) <= 0:
            return False
        if self.x2 >= shape[1]:
            return False
        if self.y2 >= shape[0]:
            return False
        return True

    def clip_by_shape(self, shape):
        """
        Clip xs and ys to be valid coordinates inside shape

        Args:
            shape: int [h, w] or None.
        """
        self.x1 = np.clip(self.x1, 0, shape[1] - 1)
        self.x2 = np.clip(self.x2, 0, shape[1] - 1)
        self.y1 = np.clip(self.y1, 0, shape[0] - 1)
        self.y2 = np.clip(self.y2, 0, shape[0] - 1)

    def roi(self, img):
        assert self.is_valid_box(img.shape[:2]), "{} vs {}".format(self, img.shape[:2])
        return img[self.y1:self.y2 + 1, self.x1:self.x2 + 1]


class FloatBox(BoxBase):
    def __init__(self, x1, y1, x2, y2):
        for k in [x1, y1, x2, y2]:
            assert isinstance(k, float), "type={},value={}".format(type(k), k)
        super(FloatBox, self).__init__(x1, y1, x2, y2)

    @property
    def w(self):
        return self.x2 - self.x1

    @property
    def h(self):
        return self.y2 - self.y1

    @staticmethod
    def from_intbox(intbox):
        return FloatBox(intbox.x1, intbox.y1,
                        intbox.x2 + 1, intbox.y2 + 1)

    def clip_by_shape(self, shape):
        self.x1 = np.clip(self.x1, 0, shape[1])
        self.x2 = np.clip(self.x2, 0, shape[1])
        self.y1 = np.clip(self.y1, 0, shape[0])
        self.y2 = np.clip(self.y2, 0, shape[0])




def draw_det_box(img, boxes, class_name, color=(255,0,0)):
    #boxes, shape:(4,)
    height, width, channels = img.shape
    images_cv = np.copy(img)

    min_x = boxes[0]
    min_y = boxes[1]
    max_x = boxes[2]
    max_y = boxes[3]

    min_x = float(min_x) * width  # normalize 1
    max_x = float(max_x) * width
    min_y = float(min_y) * height
    max_y = float(max_y) * height

    floatBox = FloatBox(min_x, min_y, max_x, max_y)
    floatBox.clip_by_shape((width, height))
    images_cv = draw_boxes(images_cv,
                           np.reshape(np.asarray([floatBox.x1, floatBox.y1, floatBox.x2, floatBox.y2]), (1, 4)),
                           color=color, labels=[class_name])

    return images_cv

def draw_text(img, pos, text, color, font_scale=0.5):
    """
    Draw text on an image.
    Args:
        pos (tuple): x, y; the position of the text
        text (str):
        font_scale (float):
        color (tuple): a 3-tuple BGR color in [0, 255]
    """
    img = img.astype(np.uint8)
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((text_w, text_h), _) = cv2.getTextSize(text, font, font_scale, 1)
    # Place text background.
    if x0 + text_w > img.shape[1]:
        x0 = img.shape[1] - text_w
    if y0 - int(1.15 * text_h) < 0:
        y0 = int(1.15 * text_h)
    back_topleft = x0, y0 - int(1.3 * text_h)
    back_bottomright = x0 + text_w, y0
    cv2.rectangle(img, back_topleft, back_bottomright, color, -1)
    # Show text.
    text_bottomleft = x0, y0 - int(0.25 * text_h)
    cv2.putText(img, text, text_bottomleft, font, font_scale, (222, 222, 222), lineType=cv2.LINE_AA)
    return img


def draw_boxes(im, boxes, labels=None, color=None, thickness=2):
    """
    Args:
        im (np.ndarray): a BGR image in range [0,255]. It will not be modified.
        boxes (np.ndarray): a numpy array of shape Nx4 where each row is [x1, y1, x2, y2].
        labels: (list[str] or None)
        color: a 3-tuple BGR color (in range [0, 255])
    Returns:
        np.ndarray: a new image.
    """
    boxes = np.asarray(boxes, dtype='int32')
    if labels is not None:
        assert len(labels) == len(boxes), "{} != {}".format(len(labels), len(boxes))
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    sorted_inds = np.argsort(-areas)    # draw large ones first
    assert areas.min() > 0, areas.min()
    # allow equal, because we are not very strict about rounding error here
    assert boxes[:, 0].min() >= 0 and boxes[:, 1].min() >= 0 \
        and boxes[:, 2].max() <= im.shape[1] and boxes[:, 3].max() <= im.shape[0], \
        "Image shape: {}\n Boxes:\n{}".format(str(im.shape), str(boxes))

    im = im.copy()
    if color is None:
        color = (15, 128, 15)
    if im.ndim == 2 or (im.ndim == 3 and im.shape[2] == 1):
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for i in sorted_inds:
        box = boxes[i, :]
        if labels is not None:
            im = draw_text(im, (box[0], box[1]), labels[i], color=color)
        cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]),
                      color=color, thickness=thickness)
    return im



if __name__ == '__main__':
    x = IntBox(2, 1, 3, 3)
    img = np.random.rand(3, 3)
    print(img)