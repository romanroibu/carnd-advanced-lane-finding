from .image import *
import numpy as np
import cv2

class Camera:

    def __init__(self):
        self.objpoints = [] # 3D points in real world space
        self.imgpoints = [] # 2D points in image plane
        pass

    def calibrate(self, images, pattern_size):

        self.objpoints = []
        self.imgpoints = []

        nx, ny = pattern_size[0], pattern_size[1]

        for image in images:

            # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (nx-1,ny-1,0)
            objp = np.zeros((nx*ny, 3), np.float32)
            objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) #x,y coordinates ????????

            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

            # If chessboard are found, add object points and image points
            if ret != True:
                continue

            self.imgpoints.append(corners)
            self.objpoints.append(objp)

    def undistort(self, image):

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, image.shape[1::-1], None, None)

        output = cv2.undistort(image, mtx, dist, None, mtx)

        return output

