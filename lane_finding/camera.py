from .image import *

import typing as t
import numpy as np
import cv2

class Camera:

    def __init__(self):
        self.objpoints = [] # 3D points in real world space
        self.imgpoints = [] # 2D points in image plane
        pass

    def calibrate(self, images: t.List[Image], pattern_size: t.Tuple[int, int]):

        self.objpoints = []
        self.imgpoints = []

        nx, ny = pattern_size[0], pattern_size[1]

        for image in images:

            # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (nx-1,ny-1,0)
            objp = np.zeros((nx*ny, 3), np.float32)
            objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

            gray_data = image.gray()._data

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray_data, (nx,ny), None)

            # If chessboard are found, add object points and image points
            if ret != True:
                continue

            self.imgpoints.append(corners)
            self.objpoints.append(objp)

    def undistort(self, image: Image):

        input_data  = image._data
        image_shape = image._data.shape

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, image_shape[1::-1], None, None)

        output_data = cv2.undistort(input_data, mtx, dist, None, mtx)

        return type(image)(output_data)
