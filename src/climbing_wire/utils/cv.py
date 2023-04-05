"""Utility functions for opencv."""

from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


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
    plt.imshow(img1)
    plt.show()


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
