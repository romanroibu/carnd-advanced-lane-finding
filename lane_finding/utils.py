from .image import Image

import typing as t
import matplotlib.pyplot as plt
from IPython.display import HTML


def show_images(titles_and_images: t.List[t.Tuple[str, Image]],
                save_path: str = None):

    # TODO: Improve implementation to work for arguments:
    # `[Image]`, `(str, Image)`, `str, Image`, `Image`

    n = len(titles_and_images)

    if n < 1:
        return

    fig, axs = plt.subplots(1, n, figsize=(24, 9))
    fig.tight_layout()

    if n == 1:
        axs = [axs]

    for i in range(n):
        ax, (title, image) = axs[i], titles_and_images[i]
        ax.imshow(image.rgb()._data)
        ax.set_title(title, fontsize=50)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    if save_path is not None:
        fig.savefig(save_path)


def show_video(fpath, width=960, height=540):
    video_html = """
        <video width="{width}" height="{height}" controls>
            <source src="{src}">
        </video>
    """.format(src=fpath, width=width, height=height)
    return HTML(video_html)
