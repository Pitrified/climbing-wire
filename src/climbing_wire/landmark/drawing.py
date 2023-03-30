"""Drawing utilities for landmarks."""

from typing import List, Mapping, Optional, Tuple

import cv2 as cv
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.pose as mp_pose
import numpy as np
from loguru import logger as lg

from climbing_wire.landmark.landmark_list import LandmarkListImg
from climbing_wire.utils.mediapipe import get_spec_from_map


def draw_landmarks(
    image: np.ndarray,
    landmarks: LandmarkListImg,
    pose_connections: List[Tuple[int, int]] | None = None,
    landmark_drawing_spec: mp_drawing.DrawingSpec
    | Mapping[int, mp_drawing.DrawingSpec]
    | None = mp_drawing.DrawingSpec(color=mp_drawing.RED_COLOR),
    connection_drawing_spec: mp_drawing.DrawingSpec
    | Mapping[Tuple[int, int], mp_drawing.DrawingSpec]
    | None = mp_drawing.DrawingSpec(),
) -> None:
    """Draw the landmarks and the connections on an image.

    Adapted from mediapipe's drawing_utils.

    Args:
        image: A three channel BGR image represented as numpy ndarray.
        landmarks: A list of landmarks to draw, already in image coordinates.
        pose_connections: A list of landmark index tuples that specifies how
            landmarks to be connected in the drawing. If this argument is
            explicitly set to None, no landmark connections will be drawn.
        landmark_drawing_spec: Either a DrawingSpec object or a mapping from hand
            landmarks to the DrawingSpecs that specifies the landmarks' drawing
            settings such as color, line thickness, and circle radius. If this
            argument is explicitly set to None, no landmarks will be drawn.
        connection_drawing_spec: Either a DrawingSpec object or a mapping from hand
            connections to the DrawingSpecs that specifies the connections' drawing
            settings such as color and line thickness. If this argument is explicitly
            set to None, no landmark connections will be drawn.

    Raises:
        ValueError: If one of the followings:
            a) If the input image is not three channel BGR.
            b) If any connection contains invalid landmark index.
    """
    if image.shape[2] != mp_drawing._BGR_CHANNELS:
        raise ValueError("Input image must contain three channel bgr data.")
    # check that the image and the landmarks are coherent ?
    if not image.shape[:2] == landmarks.img_shape[:2]:
        # raise ValueError("Input image must match landmark image.")
        # I mean you can plot the landmark somewhere else, but the de-normalization will be useless
        lg.warning(f"Input image should match landmark image.")

    num_landmarks = len(landmarks)

    # Draws the connections if the start and end landmarks are both visible.
    if pose_connections is not None and connection_drawing_spec is not None:
        for connection in pose_connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f"Landmark index is out of range. Invalid connection "
                    f"from landmark #{start_idx} to landmark #{end_idx}."
                )
            if not (landmarks.drawable[start_idx] & landmarks.drawable[end_idx]):
                continue
            # get the drawing spec for this connection if the drawing specs are a mapping
            drawing_spec = get_spec_from_map(connection_drawing_spec, connection)
            cv.line(
                image,
                landmarks.landmarks_img[start_idx],
                landmarks.landmarks_img[end_idx],
                drawing_spec.color,
                drawing_spec.thickness,
            )

    # Draws landmark points after finishing the connection lines,
    # which is aesthetically better.
    if landmark_drawing_spec is not None:
        for idx, landmark_px in enumerate(landmarks.landmarks_img):
            if not landmarks.drawable[idx]:
                continue
            drawing_spec = get_spec_from_map(landmark_drawing_spec, idx)
            # White circle border
            circle_border_radius = max(
                drawing_spec.circle_radius + 1, int(drawing_spec.circle_radius * 1.2)
            )
            cv.circle(
                image,
                landmark_px,
                circle_border_radius,
                mp_drawing.WHITE_COLOR,
                drawing_spec.thickness,
            )
            # Fill color into the circle
            cv.circle(
                image,
                landmark_px,
                drawing_spec.circle_radius,
                drawing_spec.color,
                drawing_spec.thickness,
            )
