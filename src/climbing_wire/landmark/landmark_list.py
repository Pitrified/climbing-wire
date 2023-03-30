"""A LandmarkList as numpy arrays."""

from mediapipe.framework.formats import landmark_pb2

import numpy as np

from climbing_wire.utils.mediapipe import (
    are_valid_normalized_points,
    normalized_to_pixel_coordinates,
)


class LandmarkListNp:
    """A LandmarkList as numpy arrays.

    mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
    https://github.com/google/mediapipe/blob/master/mediapipe/framework/formats/landmark.proto
    """

    def __init__(
        self,
        pose_landmarks: landmark_pb2.NormalizedLandmarkList,
    ) -> None:
        """Create the LandmarkListNp.

        Args:
            pose_landmarks (landmark_pb2.NormalizedLandmarkList): The landmarks.
        """
        # unpack the landmark data
        landmarks_norm_ls = []
        visibility_ls = []
        for landmark in pose_landmarks.landmark:  # type: ignore # the NamedTuple complains
            landmarks_norm_ls.append((landmark.x, landmark.y))
            visibility_ls.append((landmark.visibility))
        self.landmarks_norm = np.array(landmarks_norm_ls)
        self.visibility = np.array(visibility_ls)

    def __len__(self) -> int:
        """Return the number of landmarks in the list."""
        return len(self.landmarks_norm)


class LandmarkListImg(LandmarkListNp):
    """A LandmarkList as numpy arrays, with info to draw the landmarks on the source image."""

    def __init__(
        self,
        pose_landmarks: landmark_pb2.NormalizedLandmarkList,
        img_shape: tuple[int, int],
        visibility_threshold: float = 0.5,
    ) -> None:
        """Create the LandmarkListImg.

        Args:
            pose_landmarks (landmark_pb2.NormalizedLandmarkList): The landmarks.
            img_shape (tuple[int, int]): Size of the source image.
            visibility_threshold (float, optional): Minimum visibility value for a landmark.
                Defaults to 0.5.
        """
        # convert the landmarks into np arrays
        super().__init__(pose_landmarks)

        # save the configs
        self.img_shape = img_shape
        self.visibility_threshold = visibility_threshold

        # transform the landmarks in image coordinates
        self.landmarks_img = normalized_to_pixel_coordinates(
            self.landmarks_norm, self.img_shape
        )

        # decide if the points are drawable
        # must be visible and inside the image
        self.drawable = self.visibility > self.visibility_threshold
        self.drawable &= are_valid_normalized_points(self.landmarks_norm)
