"""Load a video from a file or a camera.

https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
"""

from itertools import pairwise
from pathlib import Path
from typing import Generator

import cv2 as cv
from loguru import logger as lg
import numpy as np

from climbing_wire.video.frame import Frame


def pairwise_video_frames(
    in_vid_path: Path,
    msec_interval: int = 0,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
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
    return next(iterate_video_frames(in_vid_path))[0]


def iterate_video_frames(
    in_vid_path: Path,
    msec_interval: int = 0,
    # keep_every_nth_frame: int = 1,
) -> Generator[np.ndarray, None, None]:
    """Extract frames from video, yield them one at a time.

    Args:
        in_vid_path: Input video file.
        msec_interval: Interval between frames in milliseconds.
    """
    # start capturing the feed
    cap = cv.VideoCapture(str(in_vid_path))
    # moving the cap definition outside the try block lets it be used in the
    # finally block, but if something goes wrong before we even get to the try
    # block, then we never get to the finally block ? mmm

    try:
        # get the frame rate
        fps = cap.get(cv.CAP_PROP_FPS)
        # lg.info(f"Video frame rate: {fps}")
        tot_frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
        # lg.info(f"Total number of frames: {tot_frame_count}")

        # start loading the video
        count = 0
        success = True
        while success:
            # skip some frames
            if msec_interval > 0:
                cap.set(cv.CAP_PROP_POS_MSEC, (count * msec_interval))
                pos_msec = cap.get(cv.CAP_PROP_POS_MSEC)
                # lg.debug(f"Video position after set: {pos_msec} msec")

            # extract the frame
            success, frame = cap.read()
            if not success:
                break

            pos_msec = cap.get(cv.CAP_PROP_POS_MSEC)
            # lg.debug(f"Video position: {pos_msec} msec")
            pos_usec = int(pos_msec * 1000)
            # lg.debug(f"Video position: {pos_usec:0>10} μsec")

            # # if we have an interval, skip frames while we're behind
            # if msec_interval > 0:
            #     if pos_msec < (count * msec_interval):
            #         # lg.debug(f"Skipping frame")
            #         continue

            # # or if we have a keep-every-nth-frame, skip frames
            # if count % keep_every_nth_frame == 0:
            #     lg.debug(f"Yielding frame {count}")
            #     yield frame, pos_usec

            # lg.debug(f"Yielding frame {count}")
            yield frame
            count = count + 1

    finally:
        # close the feed
        # lg.info(f"Closing video feed")
        cap.release()


def iterate_video_frames_with_timestamp(
    in_vid_path: Path,
    keep_every_nth_frame: int = 1,
) -> Generator[tuple[np.ndarray, int], None, None]:
    """Extract frames from video, yield them with a μsec timestamp.

    Args:
        in_vid_path: Input video file.
        keep_every_nth_frame: Keep every nth frame.

    Yields:
        # Frame and timestamp in μsec.
    """
    # start capturing the feed
    cap = cv.VideoCapture(str(in_vid_path))
    # moving the cap definition outside the try block lets it be used in the
    # finally block, but if something goes wrong before we even get to the try
    # block, then we never get to the finally block ? mmm

    # if you really want a msec_interval,
    # we could compute the keep_every_nth_frame from that and the fps
    # and if we need to skip many frames we could use the set(CAP_PROP_POS_MSEC)
    # (at that point a small change in frame interval does not matter)

    try:
        count = 0
        success = True
        while success:
            # extract the frame
            success, frame = cap.read()
            if not success:
                break

            # get the timestamp
            pos_msec = cap.get(cv.CAP_PROP_POS_MSEC)
            pos_usec = int(pos_msec * 1000)

            # skip frames
            if count % keep_every_nth_frame == 0:
                # lg.debug(f"Yielding frame {count}")
                yield frame, pos_usec

            count = count + 1

    finally:
        # close the feed
        # lg.info(f"Closing video feed")
        cap.release()


def list_video_frames_with_timestamp(
    in_vid_path: Path,
    keep_every_nth_frame: int = 1,
    max_frame_count: int = 0,
) -> list[Frame]:
    """Extract frames from video, return them as a list."""
    frames: list[Frame] = []

    # we need the fps for this, so we might need to open the cap here?
    # fps = 30
    # print(
    #     f"will keep a frame every {keep_every_nth_frame * fps} milliseconds,"
    #     f" will load {fps * keep_every_nth_frame * max_num_frames} milliseconds"
    # )

    frame_num = 0
    for frame, usec in iterate_video_frames_with_timestamp(
        in_vid_path,
        keep_every_nth_frame=keep_every_nth_frame,
    ):
        f = Frame(frame=frame, usec=usec, idx=frame_num)
        frames.append(f)

        frame_num += 1
        if max_frame_count > 0 and frame_num >= max_frame_count:
            break

    return frames
