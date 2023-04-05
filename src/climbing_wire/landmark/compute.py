"""Wrapper for MediaPipe Pose to compute landmarks."""
from typing import Any, Self

import cv2 as cv
from mediapipe.framework.formats import landmark_pb2
import mediapipe.python.solutions.pose as mp_pose
import numpy as np

from climbing_wire.landmark.landmark_list import LandmarkListImg


class PoseImg:
    """Wrapper of MediaPipe Pose, to work in image coordinates.

    MediaPipe Pose processes an RGB image and returns pose landmarks on the most
    prominent person detected.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
        smooth_segmentation: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        visibility_threshold: float = 0.5,
    ) -> None:
        """Initialize a PoseImg object.

        Args:
        static_image_mode: Whether to treat the input images as a batch of static
            and possibly unrelated images, or a video stream. See details in
            https://solutions.mediapipe.dev/pose#static_image_mode.
        model_complexity: Complexity of the pose landmark model: 0, 1 or 2. See
            details in https://solutions.mediapipe.dev/pose#model_complexity.
        smooth_landmarks: Whether to filter landmarks across different input
            images to reduce jitter. See details in
            https://solutions.mediapipe.dev/pose#smooth_landmarks.
        enable_segmentation: Whether to predict segmentation mask. See details in
            https://solutions.mediapipe.dev/pose#enable_segmentation.
        smooth_segmentation: Whether to filter segmentation across different input
            images to reduce jitter. See details in
            https://solutions.mediapipe.dev/pose#smooth_segmentation.
        min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for person
            detection to be considered successful. See details in
            https://solutions.mediapipe.dev/pose#min_detection_confidence.
        min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
            pose landmarks to be considered tracked successfully. See details in
            https://solutions.mediapipe.dev/pose#min_tracking_confidence.
        visibility_threshold: Minimum visibility value for a landmark.
        """
        self.pose = mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.visibility_threshold = visibility_threshold

    def __call__(
        self,
        image: np.ndarray,
        visibility_threshold: float | None = None,
    ) -> LandmarkListImg | None:
        """Compute the landmarks in image coordinates.

        Args:
            image (np.ndarray): The image.
            visibility_threshold (float | None): Minimum visibility value for a landmark.
                Set to None to use the default visibility threshold.
                Defaults to None.

        Returns:
            LandmarkListImg | None: The landmarks in image coordinates.
        """
        visibility_threshold = (
            visibility_threshold
            if visibility_threshold is not None
            else self.visibility_threshold
        )
        lms = compute_landmarks(image, self.pose)
        if lms is None:
            return None
        lli = LandmarkListImg(lms, image.shape[:2], visibility_threshold)
        return lli

    def __del__(self) -> None:
        """Close the pose object."""
        self.pose.close()

    def close(self) -> None:
        """Close the pose object."""
        self.__del__()

    def __enter__(self) -> Self:
        """Enter the context."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context."""
        self.__del__()

    def __repr__(self) -> str:
        """Return the representation of the object."""
        return f"{self.__class__.__name__}"

    def __str__(self) -> str:
        """Return the string representation of the object."""
        return self.__repr__()


def compute_landmarks(
    image: np.ndarray,
    pose: mp_pose.Pose | None = None,
    **kwargs,
) -> landmark_pb2.NormalizedLandmarkList | None:
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
