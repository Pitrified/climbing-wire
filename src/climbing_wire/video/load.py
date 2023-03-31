"""Load a video from a file or a camera.

https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
"""

from itertools import pairwise
from pathlib import Path
from typing import Generator, Tuple
import cv2 as cv
import numpy as np


def pairwise_video_frames(
    in_vid_path: Path,
    msec_interval: int = 0,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Pairwise frames from video.

    Args:
        in_vid_path: Input video file.
        msec_interval: Interval between frames in milliseconds.
    """
    yield from pairwise(iterate_video_frames(in_vid_path, msec_interval))


def first_video_frame(
    in_vid_path: Path,
) -> np.ndarray:
    """First frame from video.

    Args:
        in_vid_path: Input video file.
    """
    return next(iterate_video_frames(in_vid_path))


def iterate_video_frames(
    in_vid_path: Path,
    msec_interval: int = 0,
) -> Generator[np.ndarray, None, None]:
    """Extract frames from video, save them as .jpeg in output dir.

    Args:
        in_vid_path: Input video file.
        msec_interval: Interval between frames in milliseconds.
    """
    # start capturing the feed
    cap = cv.VideoCapture(str(in_vid_path))

    # start loading the video
    count = 0
    success = True
    while success:
        # skip some frames
        if msec_interval > 0:
            cap.set(cv.CAP_PROP_POS_MSEC, (count * msec_interval))

        # extract the frame
        success, frame = cap.read()
        if not success:
            break

        yield frame
        count = count + 1

    # close the feed
    cap.release()
