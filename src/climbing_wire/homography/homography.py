"""Compute homography between images.

https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
"""


import math
import pathlib

import cv2 as cv
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt


def compute_homography(
    img1: np.ndarray,
    img2: np.ndarray,
) -> np.ndarray:
    """Compute the homography matrix between two images."""
    # convert to grayscale
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # set up the matcher and match
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # compute the homography
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        # get the points that are good matches
        src_pts_ls = [kp1[m.queryIdx].pt for m in good]
        src_pts = np.array(src_pts_ls, dtype=np.float32).reshape(-1, 1, 2)
        dst_pts_ls = [kp2[m.trainIdx].pt for m in good]
        dst_pts = np.array(dst_pts_ls, dtype=np.float32).reshape(-1, 1, 2)
        # compute the homography with ransac
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    else:
        # might be the time to whip out LoFTR
        raise ValueError("Not enough matches found.")

    return M
