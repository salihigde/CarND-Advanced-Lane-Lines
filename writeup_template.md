## Advanced Lane Finding Writeup

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

[chess_distorted]: ./camera_cal/calibration1.jpg "Distorted"
[chess_undistorted]: ./camera_cal/output_calibration1.jpg "Undistorted"
[image_undistorted]: ./output_images/undistorted_1.jpg "Road Undistorted"
[image_thresholded]: ./output_images/thresholded_output_5.jpg "Road Thresholded"
[image_orig_6]: ./test_images/test6.jpg "Original Image"
[image_warped_6]: ./output_images/orig_warped_img_6.jpg "Warped Image"
[image_warped]: ./output_images/colored_warped_6.jpg "Thresholded Warped"
[image_mapped]: ./output_images/minv_output_warped_6.jpg "Mapped Image"
[video1]: ./video_outputs/output_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file called `camera-calibration.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

After looping for all images and collecting the object and image points I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][chess_distorted] ![alt text][chess_undistorted]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

After calling `cv2.calibrateCamera()` function I get back the calibration matrix and the distortion coefficients and save it as a pickle file to use it for undistorting the test images. For undistorting an image I call `cv2.undistort` function at line 37 (inside `img-video-generation.py`).

One of the undistorted image can be found below.
![alt text][image_undistorted]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps inside `img-video-generation.py` at lines 55 through 63 in `thresholded_img_pipeline` function).  Here's an example of my output for this step. All helper functions regarding thresholding can be found inside `tutil.py` file.

Most challenging image was the one in below because of the shadows of the trees it was hard to find proper filter to remove all the whites which were located on the road.

![alt text][image_thresholded]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes in a function called `perspectiveTransform()`, which appears in lines 66 through 112 in the file `img-video-generation.py`.  The `perspectiveTransform()` function takes as inputs an image (`image`).  I chose the hardcode the source and destination points by doing an approximate calculation in the following manner:

```python
mid_top_dist = .04
mid_bottom_dist = .355
height_percentage = .635
crop_from_bottom = .935
src = np.float32([
    [width*(0.5-mid_bottom_dist),height*crop_from_bottom],
    [width*(0.5-mid_top_dist), height*height_percentage],
    [width*(0.5+mid_top_dist), height*height_percentage],
    [width*(0.5+mid_bottom_dist),height*crop_from_bottom],
])
dst = np.float32([
    src[0]+offset,
    new_top_left+offset,
    new_top_right-offset,
    src[3]-offset
])
```

In addition to finding source and destination points I made `1.6 times` bigger portion of the trapezoid points to get a region of interest as below. Purpose of this was to reduce the noice which was inside the warped images

```python
increase_width_times = 1.6
    vertices = np.array([[
        [width*(0.5-mid_bottom_dist*increase_width_times), height*crop_from_bottom],
        [width*(0.5-mid_top_dist*increase_width_times), height*height_percentage],
        [width*(0.5+mid_top_dist*increase_width_times), height*height_percentage],
        [width*(0.5+mid_bottom_dist*increase_width_times), height*crop_from_bottom],
    ]], dtype=np.int32)

    image = tutil.region_of_interest(image, vertices)
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 185.6, 673.2  | 305.6, 673.2  |
| 588.8, 457.2  | 305.6,   0.   |
| 691.2, 457.2  | 305.6,   0.   |
| 1094.4,673.2  | 974.4, 673.2  |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image_orig_6] ![alt text][image_warped_6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I used Sliding Window algorithm, at `find_lane_pixels` function (inside `detectlines.py`) I defined `9` windows with margin of `100px` and min pixel amount of `50` each to find the `leftx, lefty, rightx` and `righty`. After finding the points, in `fit_polynomial` function at line 107 to 122, I used `A*y**2 + B*y + C` formula for both left and right lane points to fit my lane with a 2nd order polynomial kinda like this:

_Note: After trying for a while I couldn't remove the noise which can be seen in top left of the image_

![alt text][image_warped]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this at `measure_curvature_real` function in lines 145 through 153 in my code in `detectlines.py`

The function gets warped image size as a parameter. `xm_per_pix` and `ym_per_pix` are used for converting pixel to meters in real world. After finding left and right fitted points I use the curvature function to calculate left and right curvature separately and return the results by rounding them. Finally I took the average of left and right curvature to calculate the curvature from middle of the road.

The function `add_radius_and_distance_to_img` in line 31 in my code in `detectlines.py` I took the x and y points that are near to car, in the edge of road lines. After finding that points I just took the average to find the center point of the car.

The function `add_radius_and_distance_to_img` in line 32 in my code in `detectlines.py` I used camera `center_center` points with the center of the image points to determine how far the car is from the center. In that line I multiplied the result by `xm_per_pix` to find real environment results and multiplied by `1000` to find result in `cm`


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 20 through 23 in my code in `detectlines.py` in the function `process_line_detection()`.  Here is an example of my result on a test image:

![alt text][image_mapped]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./videos_output/output_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I spend a lot time on two topics. First one was the thresholding, finding the correct gradient and color thresholding combination. Second one was finding source and destination points for warping the image. Finding source and destination points took time because firstly I was trying to find the point dynamically instead of static ones. Since it was not easy to determine the trapezoid dynamically, I had to use static points for find the area to warp image bird view.

On the other hand, since I am new to python and opencv itself it is time consuming to do research on python syntax and opencv functions usage description.

Finally, in this project I picked to do the project in my local machine without using Jupyter Notebook, thats why it took time to install Anaconda and setup my VS Code to start developing.

Because of the reasons I mentioned above, I didn't have time for following;

1) Skipping the sliding windows step and not running that for each image in the video.
2) Finding the trapezoid for warping image more effectively.
3) Smoothing the lane detection to prevent line jumping

Finally, for improvement point in my project, I would have loved to work in [this article](https://airccj.org/CSCP/vol5/csit53211.pdf) and try to determine the curves more effectively in the shades or faded lanes.
