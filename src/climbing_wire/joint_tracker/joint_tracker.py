"""JointTracker tracks the joints of a person in a video stream.

A joint might be a landmark or the mean of multiple landmarks.
"""


from typing import Any

from loguru import logger as lg
import numpy as np

from climbing_wire.homography.homography import compute_homography
from climbing_wire.joint_tracker.joint_hist import JointHist
from climbing_wire.landmark.compute import PoseImg
from climbing_wire.landmark.landmark_list import LandmarkListImg
from climbing_wire.utils.mediapipe import JOINT_NAMES, JOINT_NAMES_TYPE


class JointTracker:
    """JointTracker tracks the joints of a person in a video stream.

    Track the history of several joints across multiple frames.
    """

    def __init__(
        self,
        joint_names: tuple[JOINT_NAMES_TYPE] = JOINT_NAMES,
        pose_img_kwargs: dict[str, Any] = {},
    ) -> None:
        """Create the JointTracker."""
        self.joint_names = joint_names

        # initialize the tracks for each joint
        self.joint_hists: dict[JOINT_NAMES_TYPE, JointHist] = {}
        for joint_name in joint_names:
            self.joint_hists[joint_name] = JointHist(joint_name)

        # initialize the pose estimator
        self.pose_img = PoseImg(**pose_img_kwargs)

    def process_frame_pair(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> None:
        """Process a pair of frames."""
        self.M = compute_homography(frame1, frame2)
        self.landlist = self.pose_img(frame2)
        if self.landlist is None:
            lg.warning("No landmarks found in frame.")
            return
        self.add_frame(self.landlist, self.M)

    def add_frame(
        self,
        landlist: LandmarkListImg,
        M: np.ndarray,
    ) -> None:
        """Add a new frame to the history."""
        for joint_name in JOINT_NAMES:
            self.joint_hists[joint_name].add_frame(landlist, M)
