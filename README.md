# 🛣 Advanced Lane Finding

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## 👀 Overview

The goals / steps of this project are the following:

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## 🐾 Steps

[//]: # (References)
[undistorted]: output/images/undistorted.jpg "Undistorted Sample Image"
[transformed]: output/images/transformed.jpg "Road Transformed Image"
[thresholded]: output/images/thresholded.jpg "Thresholded Binary Image"
[warped]: output/images/warped.jpg "Warped Image"
[detected]: output/images/detected.jpg "Detected Lanes Image"
[output_image]: output/images/overlaid.jpg "Final Output Image"
[project_video]: output/videos/project.mp4 "Project Video"
[challenge_video]: output/videos/challenge.mp4 "Challenge Video"
[harder_challenge_video]: output/videos/harder_challenge.mp4 "Harder Challenge Video"

### ⚙️ Camera Calibration

##### TODO:
- [ ] Briefly state how you computed the camera matrix and distortion coefficients.
- [ ] Provide an example of a distortion corrected calibration image.
- [ ] Provide an example of a distortion corrected test image.

![][undistorted]

![][transformed]

### 🎨 Color Transforms

##### TODO:
- [ ] Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.
- [ ] Provide an example of a binary image result.

![][thresholded]

### 🤔 Perspective Transform

##### TODO:
- [ ] Describe how (and identify where in your code) you performed a perspective transform
- [ ] Provide an example of a transformed image.

![][warped]

### 🧐 Lane Detection

##### TODO:
- [ ] Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

![][detected]

### ↩️ Curvature and Position Calculation

##### TODO:
- [ ] Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

## 🚘 Final Result

### 📷 Image Pipeline

##### TODO:
- [ ] Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

![][output_image]

### 📹 Video Pipeline

##### TODO:
- [ ] Show a GIF of the final video.
- [ ] Provide a link to your final video output.

![][project_video]

## 🚧 Limitations

##### TODO:
- [ ] Briefly discuss any problems / issues you faced in your implementation of this project.
- [ ] Where will your pipeline likely fail?
- [ ] What could you do to make it more robust?

