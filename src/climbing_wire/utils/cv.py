"""Utility functions for opencv."""

from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from climbing_wire.homography.homography import perspective_transform
from climbing_wire.video.frame import Frame


def cv_imshow(
    img: np.ndarray,
    ax: plt.Axes | None = None,
) -> None:
    """Show a BGR img properly.

    If ax is not None, then the image is plotted on the given axis,
    and the function returns without showing the image with plt.show().
    """
    # TODO auto detect image type (if it's grayscale)
    # img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img1 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    if ax is not None:
        ax.imshow(img1)
        return
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img1)
    plt.show()
    plt.close(fig)
    # TODO should close the figure or return the ax?


# def perspective_transform(
#     points: np.ndarray,
#     M: np.ndarray,
# ) -> np.ndarray:
#     """Transform a set of points using a given 3x3 transformation matrix.
#     Args:
#         points: A numpy array of shape (N, 2) representing the input points.
#         M: A numpy array of shape (3, 3) representing the transformation matrix.
#     Returns:
#         A numpy array of shape (N, 2) representing the transformed points.
#     """
#     # reshape the input points to the format expected by cv.perspectiveTransform
#     points = points.reshape(-1, 1, 2)
#     # transform the points using the given matrix
#     transformed_points = cv.perspectiveTransform(points, M)
#     # reshape the output points back to the original format
#     transformed_points = transformed_points.reshape(-1, 2)
#     return transformed_points


def imread_fol(
    img_fol: Path,
) -> dict[str, np.ndarray]:
    """Load all images in a folder into a dictionary."""
    imgs = {}
    for img_path in img_fol.iterdir():
        imgs[img_path.name] = cv.imread(str(img_path))
    return imgs


def show_frame(
    frame: Frame,
    ax: plt.Axes | None = None,
):
    """Wrap cv_imshow, add a title and remove the ticks.

    TODO: let cv_imshow return the ax it creates?
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    cv_imshow(frame.frame, ax=ax)

    ax.set_title(f"{frame.idx} @ {frame.usec}")

    ax_params = dict(
        which="both",
        # bottom=False,
        # top=False,
        # left=False,
        # right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    )
    ax.tick_params(**ax_params)


def show_warp(
    f1: Frame,
    f2: Frame,
    M: np.ndarray,
    ax: plt.Axes | None = None,
    dist: float | None = None,
) -> None:
    """Draw the first image warped on the second.

    TODO: Let f1 be a Frame|np.ndarray, then extract as needed.
    """
    f1img = f1.frame
    f2img = f2.frame

    corners = np.array(
        [
            [0, 0],
            [f1img.shape[1], 0],
            [f1img.shape[1], f1img.shape[0]],
            [0, f1img.shape[0]],
        ],
        dtype=np.float32,
    )
    corners_warp = perspective_transform(corners, M)
    f2_ann = cv.polylines(
        f2img.copy(), [np.int32(corners_warp)], True, 255, 3, cv.LINE_AA
    )
    # blend the person frame on the empty frame
    f1_warped = cv.warpPerspective(f1img, M, f1img.shape[:2][::-1])
    f_blend = cv.addWeighted(f1_warped, 0.5, f2_ann, 0.5, 0)

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax_params = dict(
        which="both",
        # bottom=False,
        # top=False,
        # left=False,
        # right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    )
    ax.tick_params(**ax_params)
    title = f"{f1.idx}({f1.usec//1000}):{f2.idx}({f2.usec//1000})"
    if dist is not None:
        title += f":{dist:.0f}"
    ax.set_title(title)
    cv_imshow(f_blend, ax=ax)

    # plt.show()
