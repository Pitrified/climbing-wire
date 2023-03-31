"""JointTracker tracks the joints of a person in a video stream."""


from typing import Any
import numpy as np
from climbing_wire.landmark.compute import PoseImg
from climbing_wire.landmark.landmark_list import LandmarkListImg
from climbing_wire.utils.mediapipe import JOINT_NAMES, JOINT_NAMES_TYPE


class JointTracker:
    """JointTracker tracks the joints of a person in a video stream."""

    def __init__(
        self,
        joint_names: tuple[JOINT_NAMES_TYPE] = JOINT_NAMES,
        pose_img_kwargs: dict[str, Any] = {},
    ):
        """"""
        self.joint_names = joint_names
        self.pose_img = PoseImg(**pose_img_kwargs)

        # initialize the tracks for each joint
        self.tracks = {}
        for jn in self.joint_names:
            self.tracks[jn] = np.empty((0, 2), float)

    # def update(self, landlist: LandmarkListImg): # WIP
    #     """Update the tracks with the new landmarks."""
    #     for jn in self.joint_names:
    #         lm = landlist.get_landmark_for_joint(jn)
    #         if lm is None:
    #             continue
    #         self.tracks[jn] = np.append(self.tracks[jn], lm, axis=0)
