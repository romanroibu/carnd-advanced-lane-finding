from moviepy.editor import VideoFileClip
from IPython.display import HTML

class Video:

    def show(fpath, width=960, height=540):
        video_html = """
            <video width="{width}" height="{height}" controls>
                <source src="{src}">
            </video>
        """.format(src=fpath, width=width, height=height)
        return HTML(video_html)

    def read(fpath):
        clip = VideoFileClip(fpath)
        return Video(clip)

    def __init__(self, clip):
        self._clip = clip

    def subclip(start=0, end=None):
        clip = self._clip.subclip(start, end)
        return Video(clip)

    def process(f):
        clip = self._clip.fl_image(f)
        return Video(clip)

    def write(fpath, audio=False, progress_bar=False, verbose=True):
        return self._clip.write_videofile(white_output, audio=audio, progress_bar=progress_bar, verbose=verbose)

