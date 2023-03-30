"""Misc utils for landmark detection."""

from typing import List, Mapping, Tuple, TypeVar, cast

import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.pose as mp_pose
import numpy as np

T = TypeVar("T")


def normalized_to_pixel_coordinates(
    normalized_points: np.ndarray,
    image_size: Tuple[int, int],
    clip_to_image: bool = True,
) -> np.ndarray:
    """Convert normalized value pair to pixel coordinates.

    Set ``clip_to_image`` to False to allow points outside of the image.

    Vectorized version of drawing_utils._normalized_to_pixel_coordinates
    """
    image_width, image_height = image_size
    x_px = np.floor(normalized_points[:, 0] * image_width).astype(int)
    y_px = np.floor(normalized_points[:, 1] * image_height).astype(int)
    if clip_to_image:
        x_px = np.clip(x_px, 0, image_width - 1)
        y_px = np.clip(y_px, 0, image_height - 1)
    return np.column_stack((x_px, y_px))


def are_valid_normalized_points(points: np.ndarray) -> np.ndarray:
    """Check if the points are between 0 and 1.

    Vectorized version of drawing_utils.is_valid_normalized_value,
    (inside drawing_utils._normalized_to_pixel_coordinates).

    Args:
        points: Array of shape (num_points, 2).

    Returns:
        np.ndarray: Boolean array of shape (num_points,).
    """
    zeros = (points > 0) | np.isclose(points, 0)
    ones = (points < 1) | np.isclose(points, 1)
    valid = zeros & ones
    return valid.all(axis=1)


def get_spec_from_map(
    drawing_spec: mp_drawing.DrawingSpec | Mapping[T, mp_drawing.DrawingSpec],
    key: T,
) -> mp_drawing.DrawingSpec:
    """Extract a DrawingSpec from a Mapping or return the DrawingSpec itself."""
    if isinstance(drawing_spec, Mapping):
        return drawing_spec[key]
    return drawing_spec


def get_default_pose_connections() -> List[Tuple[int, int]]:
    """Get the default pose connections.

    Cast the connections to a list of tuples for the sake of type checking.
    """
    pose_connections = cast(List[Tuple[int, int]], mp_pose.POSE_CONNECTIONS)
    return pose_connections
