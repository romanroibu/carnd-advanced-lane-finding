from .image import *

import random
import typing as t
import numpy as np
from moviepy.editor import VideoFileClip

class Video:

    def read(fpath):
        clip = VideoFileClip(fpath)
        return Video(clip)

    def __init__(self, clip):
        self._clip = clip

    @property
    def frame_count(self) -> int:
        return int(self._clip.fps * self._clip.duration)

    def __getitem__(self, key: int):
        time  = float(key) / self._clip.fps
        frame = self._clip.get_frame(time)
        image = Video.__image_from_frame(frame)
        return image

    def random_frame(self) -> Image:
        index = random.randint(0, self.frame_count-1)
        return self[index]

    def subclip(self, start=0, end=None):
        clip = self._clip.subclip(start, end)
        return Video(clip)

    def process(self, f):
        def raw_f(frame):
            in_image  = Video.__image_from_frame(frame)
            out_image = f(in_image)
            return out_image._data
        clip = self._clip.fl_image(raw_f)
        return Video(clip)

    def write(self, fpath, audio=False, progress_bar=True, verbose=True):
        return self._clip.write_videofile(fpath, audio=audio, progress_bar=progress_bar, verbose=verbose)

    def __image_from_frame(frame: np.ndarray) -> Image:
        frame = np.moveaxis(frame, -1, 0)
        return RGBImage.fromChannels(frame)
        
