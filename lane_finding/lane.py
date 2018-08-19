from .image import *

import typing as t
import numpy as np


class Lane():

    def __init__(self, image: BinaryImage, x: np.ndarray, y: np.ndarray):

        image_shape = image._data.shape

        # Fit a second order polynomial
        coefficients = np.polyfit(y, x, 2)

        # Generate x and y values for plotting
        plot_y = np.linspace(0, image_shape[0]-1, image_shape[0])

        # Calc polynomial using plot_y and coefficients
        fit_x = coefficients[0] * plot_y**2 \
              + coefficients[1] * plot_y \
              + coefficients[2]

        self._x = x
        self._y = y

        self._fit_x = fit_x
        self._plot_y = plot_y
        self._coefficients = coefficients
        self._base = Point(x=0, y=0) #FIXME!!!!!!!!!!!!!

        self._curvature_radius_in_meters = None #TODO: Compute

    @property
    def x(self):
        return self._x

    @property
    def base(self) -> Point:
        return self._base

#TODO: Rename Lanes to Road
class Lanes():
    def __init__(self, image: BinaryImage, left: Lane, right: Lane):
        self.left  = left
        self.right = right

        horizontal_midpoint = (right.base.x + left.base.x) // 2
        self.center_offset = horizontal_midpoint - image.width

    @property
    def curvature_radius_in_meters(self) -> float:
        #TODO: Return the mean of the left and right curvatures
        return None

    @property
    def camera_offset_in_meters(self) -> float:
        #TODO: Compute the offset of the camera relative to the detected road
        return None
