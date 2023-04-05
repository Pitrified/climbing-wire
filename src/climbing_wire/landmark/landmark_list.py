"""A LandmarkList as numpy arrays."""

from typing import Literal, Self

from loguru import logger as lg
from mediapipe.framework.formats import landmark_pb2
import numpy as np

from climbing_wire.utils.mediapipe import (
    POSE_LANDMARKS_MAP,
    POSE_LANDMARKS_NAMES,
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
        landmarks_norm_ls: list[tuple[float, float]] = []
        visibility_ls: list[float] = []
        for landmark in pose_landmarks.landmark:  # type: ignore # the NamedTuple complains
            landmarks_norm_ls.append((landmark.x, landmark.y))
            visibility_ls.append((landmark.visibility))
        self.landmarks_norm = np.array(landmarks_norm_ls)
        self.visibility = np.array(visibility_ls)
        # so that we can create a copy of this object
        # meh
        self.pose_landmarks = pose_landmarks

    def __len__(self) -> int:
        """Return the number of landmarks in the list."""
        return len(self.landmarks_norm)

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        r = ""
        for lname, lpos in zip(POSE_LANDMARKS_NAMES, self.landmarks_norm):
            r += f"{lname:>19s} {lpos[0]:.3f} {lpos[1]:.3f} \n"
        return r


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

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        r = ""
        for lname, lpos, viz, draw in zip(
            POSE_LANDMARKS_NAMES, self.landmarks_img, self.visibility, self.drawable
        ):
            r += f"{lname:>19s} {lpos} {viz:.3f} {draw}\n"
        return r

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return (
            f"{self.__class__.__name__}"
            f"({self.drawable.sum()}/{len(self.drawable)} drawable)"
        )

    def get_landmark_for_joint(
        self,
        which_landmark: Literal[
            "left_hand",
            "right_hand",
            "left_foot",
            "right_foot",
        ],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the position of a joint, and a visibility value.

        The point will be returned as a 1x2 array.

        TODO: consider a mean of the landmarks for a joint,
            to provide a more robust estimate of the joint position.
            * left hand: "LEFT_WRIST", "LEFT_PINKY", "LEFT_INDEX", "LEFT_THUMB",
            * right hand: "RIGHT_WRIST", "RIGHT_PINKY", "RIGHT_INDEX", "RIGHT_THUMB",
            * left foot: "LEFT_ANKLE", "LEFT_HEEL", "LEFT_FOOT_INDEX",
            * right foot: "RIGHT_ANKLE", "RIGHT_HEEL", "RIGHT_FOOT_INDEX",
        """
        if which_landmark == "left_hand":
            land_idx = POSE_LANDMARKS_MAP["LEFT_WRIST"]
        elif which_landmark == "right_hand":
            land_idx = POSE_LANDMARKS_MAP["RIGHT_WRIST"]
        elif which_landmark == "left_foot":
            land_idx = POSE_LANDMARKS_MAP["LEFT_ANKLE"]
        elif which_landmark == "right_foot":
            land_idx = POSE_LANDMARKS_MAP["RIGHT_ANKLE"]

        return self.landmarks_img[land_idx : land_idx + 1, :], self.visibility[land_idx]

    def copy(self) -> Self:
        """Return a copy of the object."""
        # lg.debug("Copying LandmarkListImg.")
        return LandmarkListImg(
            self.pose_landmarks,
            self.img_shape,
            self.visibility_threshold,
        )
