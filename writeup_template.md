
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./test_images/test2.jpg "Test image"
[image1]: ./output_images/undistorted_test2.jpg "Undistorted"
[image2]: ./Debug_bad_lane_detection.jpg "bad lane detection"
[image3]: ./output_images/example_threshold_undistorted_test2.jpg "Thresholding Example"
[image4]: ./output_images/example_Warped_threshold_undistorted_test2.jpg "Warp Example"
[image5]: ./output_images/color_fit_lines.jpg "Fit Visual"
[final_image]: ./output_images/Final_test2.jpg "Output"
[project_video_marked]: ./project_video_marked.mp4 "Video"

[Advanced-Lane-Lines.ipynb]: ./Advanced-Lane-Lines.ipynb "Advanced-Lane-Lines.ipynb"
[calibration.py]: ./calibration.py "calibration module"
[lanes.py]: ./lanes.py "lanes lines helper models"
[thresholding.py]: ./thresholding.py "Thresholding module"
[pipeline.py]: ./pipeline.py "End to end pipeline module"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it!

### Summary of code pointers:
At high level all the code to test individual stages of pipeline is in [Advanced-Lane-Lines.ipynb]
It uses following modules:
[calibration.py]
[thresholding.py]
[lanes.py]
[pipeline.py]

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 2nd and 3rd code cell of the IPython notebook [Advanced-Lane-Lines.ipynb] and [calibration.py]

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in images under: ./camera_cal/*.jpg.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
##### Test image
![Test image][image0]
##### Undistorted
![Undistorted][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
Shown in above section is a test image and undistorted image.
Section 4 of [Advanced-Lane-Lines.ipynb] has all the test images and corresponding undistorted images.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image.  The code for this step is contained in the 5th code cell of the IPython notebook [Advanced-Lane-Lines.ipynb] and [thresholding.py]
Here's an example of my output for this step.  

![Thresholding Example][image3]

Cell 5 of the IPython notebook [Advanced-Lane-Lines.ipynb] has all the test examples transformed.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for this step is contained in the 6th code cell of the IPython notebook [Advanced-Lane-Lines.ipynb].

I chose the hardcode the source and destination points in the following manner:

```
offset = 200
src = np.float32([
    [  588,   446 ],
    [  691,   446 ],
    [ 1126,   673 ],
    [  153 ,   673 ]])
dst = np.float32([[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]], [offset, img_size[1]]])

```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Warp Example][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I implemented this step in 7th code cell in my code in the IPython notebook [Advanced-Lane-Lines.ipynb] and [lanes.py] detect_lanes_full() function.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I implemented this step in 8th code cell in my code in the IPython notebook [Advanced-Lane-Lines.ipynb] and [lanes.py] get_radius() function.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in 9th code cell in my code in the IPython notebook [Advanced-Lane-Lines.ipynb] and [lanes.py] plot_lanes() functioon.  Here is an example of my result on a test image:

![Final][final_image]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [project_video_marked](./project_video_marked.mp4)

---

###Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Initially my pipeline only used the detect lanes full function (where lane detection uses histogram and windowing technique every time) and this worked fairly well for project video except for few frames towards the end.  At this point there was no check for lane line detection sanity.

Then I added the incremental line detection and found that over time lane capture deteriorates, however with lack of lane lines sanity it gave me worse detection than before but much faster detection.

Then I added sanity check based on left and right lane radius of curvature with the assumption that difference of radius of curvature of the 2 lanes = width of the lane and anytime I get width < 2 m and > 5 m assume bad lane detection.  This proved to be wrong assumption and this approach was not a good way for sanity check.  Untimately I added the start of the lines are within resonable distance both min and max from center of image as hinted in lesson material as sanity check.  This worked much better.  

Also I found that instead of average line plots weighted average of line plots where more recent lines have higher weights seemed to work better.

One looking at images where lane detection did not do well. Like the one here: [image2], I realised it was becase of detected lanes were not parallel.  So I added a validation to check for parallel lanes by taking derivative of polynomil that represents the lane and looking at its value at base of image for both the lanes.

My lane detection does not do well on challenge and harder challenge videos.  I need to spend more time there.  One thought I had was to capture images and make sure thresholding is adjusted to detect those lines and avoid other high gradient lines in the video (curbs that are not lanes and markers in middle of lane).

