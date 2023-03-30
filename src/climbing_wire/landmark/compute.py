"""Wrapper for MediaPipe Pose to compute landmarks."""
from typing import Any

import cv2 as cv
import mediapipe.python.solutions.pose as mp_pose
import numpy as np


def compute_landmarks(
    image: np.ndarray,
    pose: mp_pose.Pose | None = None,
    **kwargs,
) -> Any | None:
    """Run MediaPipe Pose to compute landmarks."""
    if pose is None:
        pose = mp_pose.Pose(**kwargs)

    # To improve performance,
    # mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    if not results.pose_landmarks:  # type: ignore
        return None
    return results.pose_landmarks  # type: ignore
