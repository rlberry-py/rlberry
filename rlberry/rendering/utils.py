_FFMPEG_INSTALLED = True

import numpy as np

try:
    import ffmpeg
except:
    _FFMPEG_INSTALLED = False


def video_write(fn, images, framerate=60, vcodec='libx264'):
    """
    Save list of images to a video file.

    Source: https://github.com/kkroening/ffmpeg-python/issues/246#issuecomment-520200981 
    Modified so that framerate is given to .input(), as suggested in the thread, to avoid
    skipping frames.

    Parameters
    ----------
    fn : string
        filename 
    images : list or np.array
        list of images to save to a video.
    framerate : int
    """
    global _FFMPEG_INSTALLED

    if not _FFMPEG_INSTALLED:
        print(
            "Error: not able to save video, ffmpeg-python package required (https://github.com/kkroening/ffmpeg-python)")
        return

    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n, height, width, channels = images.shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=framerate)
            .output(fn, pix_fmt='yuv420p', vcodec=vcodec)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()
