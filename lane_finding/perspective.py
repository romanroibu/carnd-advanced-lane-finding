from .image import *
import numpy as np
import cv2

class Perspective:

    def __init__(self, src, dst):
        self._src  = np.float32(src)
        self._dst  = np.float32(dst)
        self._T    = cv2.getPerspectiveTransform(self._src, self._dst)
        self._I    = cv2.getPerspectiveTransform(self._dst, self._src)

    def transform(self, image):
        return self._warp(image, self._T)

    def inverse(self, image):
        return self._warp(image, self._I)

    def _warp(self, image, matrix):
        input_data  = image.data()
        size = image.rgb().size()
        output_data = cv2.warpPerspective(input_data, matrix, size, flags=cv2.INTER_LINEAR)
        return type(image)(output_data)


