import typing as t
import numpy as np
import cv2

# Abstract base class
class Image:

    def read(fpath):
        return BGRImage(cv2.imread(fpath))

    @classmethod
    def fromChannels(klass, channels):
        # precondition: channels is a tuple with channel data
        if len(channels) == 1:
            channels = (channels[0], channels[0], channels[0])
        return klass(np.dstack(channels))

    def __init__(self, image: np.ndarray):
        self._data = image

    def copy(self):
        return type(self)(np.copy(self._data))

    def height(self):
        return self._data.shape[0]

    def width(self):
        return self._data.shape[1]

    def size(self):
        return (self.width(), self.height())

    def binary(self, thresh):
        return self.gray().binary(thresh)

    def h_channel(self):
        return GrayImage(self.hls()._data[:,:,0])

    def l_channel(self):
        return GrayImage(self.hls()._data[:,:,1])

    def s_channel(self):
        return GrayImage(self.hls()._data[:,:,2])

    def abs_sobel_x(self, kernel=3):
        return self.abs_sobel(dx=1, dy=0, kernel=kernel)

    def abs_sobel_y(self, kernel=3):
        return self.abs_sobel(dx=0, dy=1, kernel=kernel)

    def abs_sobel(self, dx, dy, kernel):
        # Calculate directional gradient

        sobel = cv2.Sobel(self._data, cv2.CV_64F, dx, dy, ksize=kernel)

        absolute = np.absolute(sobel)

        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled = np.uint8(255 * absolute / np.max(absolute))

        return type(self)(scaled)

    def magnitude(self, kernel=3):
        # Calculate gradient magnitude

        sobelx = cv2.Sobel(self._data, cv2.CV_64F, 1, 0, ksize=kernel)
        sobely = cv2.Sobel(self._data, cv2.CV_64F, 0, 1, ksize=kernel)

        absolute = np.sqrt(sobelx**2 + sobely**2)

        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled = np.uint8(255 * absolute / np.max(absolute))

        return type(self)(scaled)

    def direction(self, kernel=15):
        # Calculate gradient direction

        sobelx = cv2.Sobel(self._data, cv2.CV_64F, 1, 0, ksize=kernel)
        sobely = cv2.Sobel(self._data, cv2.CV_64F, 0, 1, ksize=kernel)

        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)

        direction = np.arctan2(abs_sobely, abs_sobelx)

        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled = np.uint8(255 * direction / (np.pi/2))

        return type(self)(scaled)



class BGRImage(Image):

    def bgr(self):
        return BGRImage(self._data)

    def rgb(self):
        return RGBImage(cv2.cvtColor(self._data, cv2.COLOR_BGR2RGB))

    def hls(self):
        return HLSImage(cv2.cvtColor(self._data, cv2.COLOR_BGR2HLS))

    def gray(self):
        return GrayImage(cv2.cvtColor(self._data, cv2.COLOR_BGR2GRAY))



class RGBImage(Image):

    def bgr(self):
        return BGRImage(cv2.cvtColor(self._data, cv2.COLOR_RGB2BGR))

    def rgb(self):
        return RGBImage(self._data)

    def hls(self):
        return HLSImage(cv2.cvtColor(self._data, cv2.COLOR_RGB2HLS))

    def gray(self):
        return GrayImage(cv2.cvtColor(self._data, cv2.COLOR_RGB2GRAY))


class HLSImage(Image):

    def bgr(self):
        return BGRImage(cv2.cvtColor(self._data, cv2.COLOR_HLS2BGR))

    def rgb(self):
        return RGBImage(cv2.cvtColor(self._data, cv2.COLOR_HLS2RGB))

    def hls(self):
        return HLSImage(self._data)

    def gray(self):
        return GrayImage(cv2.cvtColor(self._data, cv2.COLOR_HLS2GRAY))



class GrayImage(Image):

    def bgr(self):
        return BGRImage.fromChannels((self._data,))

    def rgb(self):
        return RGBImage.fromChannels((self._data,))

    def hls(self):
        return HLSImage.fromChannels((self._data,))

    def gray(self):
        return GrayImage(self._data)

    def binary(self, thresh):
        binary_data = np.zeros_like(self._data)
        binary_data[(self._data >= thresh[0]) & (self._data <= thresh[1])] = 1
        return BinaryImage(binary_data)



class BinaryImage(Image):

    def bgr(self):
        return self.gray().bgr()

    def rgb(self):
        return self.gray().rgb()

    def hls(self):
        return self.gray().hls()

    def gray(self):
        return GrayImage(self._data * 255)



