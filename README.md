# Climbing wire

## Procedure

### Draw limb traces on the video

1. Load video
1. Compute wireframe
1. Convert landmarks from normalized to image space
1. Compute homographies between current and previous frame
1. Convert old landmarks to current frame reference system
1. Draw the landmarks on the current frame
1. Assemble the video

### Wireframe playback

1. Load the videos.
1. Align the videos:
    1. Compute homographies between the two videos' frames.
    1. Compute some distance related to the homographies.
    1. Use dynamic time warping to align the two videos.
       The empty video will be shorter: repeat frames as needed.
1. Compute wireframe on current frame.
1. Get the homography between current frame and reference.
1. Convert wireframe to reference frame.
1. Plot it.
1. Assemble the video

## Open points and ideas

* if a landmark is missing try to use the others
* how to transform a whole wireframe (LandmarkListImg)
* line thickness proportional to speed
* how to deal with sharp movement
* image -> segmentation mask ?
* should compute_homography be part of a class?
  so something like `ch = ComputeHomography(img1, img2)`
  and in ch you have all the intermediate results

* consider the framerate when using DTW

* as we can segment person/background, we could build the background by stitching it from frames where the person is not there, thus needing a single video

## Package structure

#### opencv and homography

* [x] homography computation (SIFT+FLANN+RANSAC)
* [x] perspectiveTransform (with auto reshaping, and auto cast float-int)
* [x] cv_imshow
* [ ] draw matches + polylines
* [ ] warpPerspective

#### mediapipe and landmark

* [ ] normalized_to_pixel_coordinates
* [ ] are_valid_normalized_points
* [ ] LandmarkListNp
* [ ] LandmarkListImg
* [ ] compute_landmarks (image -> wireframe)
* [ ] draw_landmarks
* [ ] get_spec_from_map

#### Video load/create

* [ ] video iterator
* [ ] video_to_frames
* [ ] frames_to_video

#### LimbTracker

* init
* add frames one by one

## Resources

### Pose tracking

* https://colab.research.google.com/drive/1uCuA6We9T5r0WljspEHWPHXCT_2bMKUy

### Homography

* https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
