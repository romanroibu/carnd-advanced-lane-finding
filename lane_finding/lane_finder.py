from .image import *
from .lane import *

import collections as c
import typing as t
import numpy as np
import cv2


class LaneFinder:

    def __init__(self, nwindows: int=9, margin: int=100, minpix: int=50,
                 x_m_per_pix: float=3.7/700, y_m_per_pix: float=30/720):

        # Number of sliding windows
        self._nwindows = nwindows
        # Width of the windows +/- margin to search
        self._margin = margin
        # Minimum number of pixels found to recenter window
        self._minpix = minpix

        #TODO: Remove
        self._left_fit = None
        self._right_fit = None

        self._lanes: Lanes = None

    # @property
    # def left_lane(self) -> Lane:
    #     return self._lanes[0]

    # @property
    # def right_lane(self) -> Lane:
    #     return self._lanes[1]

    # @property
    # def lanes(self) -> Lanes:
    #     return self._lanes

    # @lanes.setter
    # def lanes(self, value: Lanes):
    #     self._lanes = lanes

    def overlay(self, image: Image, lanes: Lanes, color=(0,255,0), search_windows: t.List[Rectangle]=[]):

        #TODO: Remove `lanes` argument and overlay smoothed lanes based on previous `n` detections

        # Create an image to draw the lines on
        output = Image.zeros_like(image.rgb())

        fit_x_left   = lanes.left._fit_x
        fit_x_right  = lanes.right._fit_x
        plot_y_left  = lanes.left._plot_y
        plot_y_right = lanes.right._plot_y

        #TODO: Review all stepps to convert Lane instance into points array for cv2.fillPoly

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([fit_x_left, plot_y_left]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_x_right, plot_y_right])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(output._data, np.int_([pts]), color)

        for rect in search_windows:
            output.draw_rectangle(rect, color=(0,255,0), thickness=2)

        return output

    def overlay_info(self, image: Image, lanes: Lanes, color=(0,255,0)):

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        x_offset = 50
        y_offset = 50
        y_spacing = 20

        # Create an image to draw the lines on
        output = Image.zeros_like(image.rgb())

        info_lines = [
            'Curvature: {} m'.format(lanes.curvature_radius_in_meters),
            'Camera offset: {} m'.format(lanes.camera_offset_in_meters)
        ]

        bottom_left = (x_offset, y_offset)

        for info in info_lines:

            (text_w, text_h), text_baseline = cv2.getTextSize(info, font_face, font_scale, thickness)
            bottom_left = (bottom_left[0], bottom_left[1] + text_h)

            cv2.putText(output._data, info, bottom_left, font_face, font_scale, color, thickness)
            bottom_left = (bottom_left[0], bottom_left[1] + y_spacing)

        return output


    def search(self, image: BinaryImage) -> t.Tuple[Lanes, t.List[Rectangle]]:

        if self._lanes is not None:
            new_lanes, search_windows = self.extend_lanes(image, previous_lanes=self._lanes), []
        else:
            new_lanes, search_windows = self.find_lanes(image)

        #TODO: Validate new_lanes compared to previously detected lanes

        #TODO: Append new_lanes to a list of previously found lanes
        self._lanes = new_lanes

        return new_lanes, search_windows

    def extend_lanes(self, image: BinaryImage, previous_lanes: Lanes) -> Lanes:

        # Grab activated pixels
        nonzero   = image._data.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        # Search based on activated x-values within the +/- margin of polynomial function

        coeff_left = previous_lanes.left._coefficients
        fit_x_left = coeff_left[0]*(nonzero_y**2) + coeff_left[1]*nonzero_y + coeff_left[2]

        coeff_right = previous_lanes.right._coefficients
        fit_x_right = coeff_right[0]*(nonzero_y**2) + coeff_right[1]*nonzero_y + coeff_right[2]

        left_indices  = ((nonzero_x > (fit_x_left  - self._margin)) & (nonzero_x < (fit_x_left  + self._margin)))
        right_indices = ((nonzero_x > (fit_x_right - self._margin)) & (nonzero_x < (fit_x_right + self._margin)))

        # Extract left line pixel positions
        left_x = nonzero_x[left_indices]
        left_y = nonzero_y[left_indices]

        # Extract right line pixel positions
        right_x = nonzero_x[right_indices]
        right_y = nonzero_y[right_indices]

        left  = Lane(image=image, x=left_x,  y=left_y)
        right = Lane(image=image, x=right_x, y=right_y)

        return Lanes(image=image, left=left,right=right)


    def find_lanes(self, image: BinaryImage) -> t.Tuple[Lanes, t.List[Rectangle]]:

        image_data  = image._data
        image_shape = image._data.shape

        # Take a histogram of the bottom half of the image
        histogram = np.sum(image_data[image_shape[0]//2:,:], axis=0)

        # Create a search window accumulator for debugging
        search_windows = []

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        x_base_left  = np.argmax(histogram[:midpoint])
        x_base_right = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(image_shape[0]//self._nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero   = image_data.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        # Current positions to be updated later for each window in nwindows
        x_current_left  = x_base_left
        x_current_right = x_base_right

        # Create empty lists to receive left and right lane pixel indices
        indices_left  = []
        indices_right = []

        # Step through the windows one by one
        for window in range(self._nwindows):

            # Identify window boundaries in x and y (and right and left)
            win_y_low  = image_shape[0] - (window+1) * window_height
            win_y_high = image_shape[0] -  window    * window_height

            # Find the four below boundaries of the window
            win_x_low_left   = x_current_left  - self._margin
            win_x_high_left  = x_current_left  + self._margin
            win_x_low_right  = x_current_right - self._margin
            win_x_high_right = x_current_right + self._margin

            # Collect the window for debugging

            search_windows.append(Rectangle((
                Point(x=win_x_low_left,  y=win_y_low),
                Point(x=win_x_high_left, y=win_y_high)
            )))
            search_windows.append(Rectangle((
                Point(x=win_x_low_right,  y=win_y_low),
                Point(x=win_x_high_right, y=win_y_high)
            )))

            # Identify the nonzero pixels in x and y within the window
            good_indices_left   = ((win_x_low_left  <= nonzero_x) & (nonzero_x < win_x_high_left) \
                                &  (win_y_low       <= nonzero_y) & (nonzero_y < win_y_high)).nonzero()[0]
            good_indices_right  = ((win_x_low_right <= nonzero_x) & (nonzero_x < win_x_high_right) \
                                &  (win_y_low       <= nonzero_y) & (nonzero_y < win_y_high)).nonzero()[0]

            # Append these indices to the lists
            indices_left.append(good_indices_left)
            indices_right.append(good_indices_right)

            # If there are more than self._minpix pixels in left/right lane,
            # recenter next window (`x_current_left`/`x_current_right`) on their mean position
            if len(good_indices_left) > self._minpix:
                x_current_left = np.int(np.mean(nonzero_x[good_indices_left]))
            if len(good_indices_right) > self._minpix:
                x_current_right = np.int(np.mean(nonzero_x[good_indices_right]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        indices_left  = np.concatenate(indices_left)
        indices_right = np.concatenate(indices_right)

        # Extract left line pixel positions
        left_x = nonzero_x[indices_left]
        left_y = nonzero_y[indices_left]

        # Extract right line pixel positions
        right_x = nonzero_x[indices_right]
        right_y = nonzero_y[indices_right]

        left  = Lane(image=image, x=left_x,  y=left_y)
        right = Lane(image=image, x=right_x, y=right_y)

        return Lanes(image=image, left=left,right=right), search_windows

    def __visualize(self, image: BinaryImage, leftx, lefty, rightx, righty, left_fitx, right_fitx, ploty):

        # Color in left and right line pixels
        image = image.rgb()
        image._data[lefty, leftx] = [255, 0, 0]
        image._data[righty, rightx] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-self._margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self._margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-self._margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self._margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        window_image = Image.zeros_like(image)
        cv2.fillPoly(window_image._data, np.int_([left_line_pts]),  (0,255,0))
        cv2.fillPoly(window_image._data, np.int_([right_line_pts]), (0,255,0))

        # Draw the polynomial lines onto the image
        left_line_middle  = np.int32([np.transpose(np.vstack([left_fitx,  ploty]))])
        right_line_middle = np.int32([np.transpose(np.vstack([right_fitx, ploty]))])

        lines_image = Image.zeros_like(image)
        cv2.polylines(lines_image._data, left_line_middle,  False, (255,255,0), 2, cv2.LINE_AA)
        cv2.polylines(lines_image._data, right_line_middle, False, (255,255,0), 2, cv2.LINE_AA)

        result_image = Image.combine(image, 1, window_image, 0.3)
        result_image = Image.combine(result_image, 1, lines_image, 1)

        return result_image

    def measure_curvature_pixels(self, left_fit, right_fit, ploty):
        '''
        Calculates the curvature of polynomial functions in pixels.
        '''

        # Calculation of R_curve (radius of curvature) in pixels
        def curveradf(x, a, b):
            return (1 + (2*a*x+b)**2)**(3/2) / np.abs(2*a)

        return self._measure_curvature(left_fit, right_fit, ploty, curveradf)

    def measure_curvature_meters(self, left_fit, right_fit, ploty, x_m_per_pix=None, y_m_per_pix=None):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        # y_m_per_pix - meters per pixel in y dimension
        # x_m_per_pix - meters per pixel in x dimension

        if x_m_per_pix is None:
            x_m_per_pix = self.x_m_per_pix

        if y_m_per_pix is None:
            y_m_per_pix = self.y_m_per_pix

        # Calculation of R_curve (radius of curvature) in meters
        def curveradf(x, a, b):
            return ((1 + (2*a*x*y_m_per_pix + b)**2)**(3/2)) / np.absolute(2*a)

        return self._measure_curvature(left_fit, right_fit, ploty, curveradf)

    def _measure_curvature(self, left_fit, right_fit, ploty, curveradf):
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        left_curverad  = curveradf(y_eval, left_fit[0],  left_fit[1])
        right_curverad = curveradf(y_eval, right_fit[0], right_fit[1])

        return left_curverad, right_curverad

    def __overlay(self, image, left_fitx, right_fitx, ploty, color=(0,255,0)):

        # Create an image to draw the lines on
        output = Image.zeros_like(image.rgb())

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(output._data, np.int_([pts]), color)

        return output
