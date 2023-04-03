"""Track the history of a joint across multiple frames.

A joint might be a landmark or the mean of multiple landmarks.
"""


import numpy as np
from climbing_wire.homography.homography import perspective_transform
from climbing_wire.landmark.landmark_list import LandmarkListImg
from climbing_wire.utils.mediapipe import JOINT_NAMES_TYPE


class JointHist:
    """Track the history of a joint across multiple frames."""

    def __init__(
        self,
        which_joint: JOINT_NAMES_TYPE,
    ) -> None:
        """Create the JointHist."""
        self.which_joint: JOINT_NAMES_TYPE = which_joint

        # prepare the joint hist data
        self.track = np.empty((0, 2), float)
        self.visibility = np.empty((0,), float)

    def add_frame(
        self,
        landlist: LandmarkListImg,
        M: np.ndarray,
    ) -> None:
        """Add a new frame to the history.

        Appending to a numpy array is slow, TODO.
        """
        # warp the old joint positions to the new frame
        self.track = perspective_transform(self.track, M)

        # get the new joint position
        lm, viz = landlist.get_landmark_for_joint(self.which_joint)
        self.track = np.append(self.track, lm, axis=0)
        self.visibility = np.append(self.visibility, viz)

    def __len__(self) -> int:
        """Return the number of frames in the history."""
        return len(self.track)
