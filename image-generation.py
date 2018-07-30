import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from detectlines import *
import tutil
from Line import *
from moviepy.editor import VideoFileClip
from IPython.display import HTML

dist_pickle = pickle.load(open( "camera_cal/my_dist_pickle.p", "rb" ))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def pipeline():
    images = glob.glob('test_images/test*.jpg')

    for i, fname in enumerate(images):
        image = cv2.imread(fname)

        result = process_image(image)
        
        warped_img_name = 'warped_' + str(i+1) + '.jpg'
        img_name = 'output_images/' + 'minv_output_' + warped_img_name  
        cv2.imwrite(img_name, result)

def process_image(image):
    thresholded_img = thresholded_img_pipeline(image)
    
    warped_img, Minv = perspectiveTransform(thresholded_img)

    params = Line()
    params.Minv = Minv
    detect_lines = DetectLines(image, warped_img, params)
    unwarped_weighted = detect_lines.process_line_detection()

    return detect_lines.add_radius_and_distance_to_img(unwarped_weighted)

def video_gen():
    white_output = 'videos_output/output_project_video.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)

def thresholded_img_pipeline(image):
    ksize = 15
    image = cv2.undistort(image, mtx, dist, None, mtx)
    gradx = tutil.abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(12, 255))
    grady = tutil.abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(25, 255))
    color_t = tutil.color_thresh(image, s_thresh=(100,255), v_thresh=(50,255))

    combined = np.zeros_like(color_t)
    combined[(((gradx == 1) & (grady == 1)) | (color_t == 1))] = 255

    return combined

def undistort():
    images = glob.glob('test_images/test*.jpg')
    dist_pickle = pickle.load( open( "camera_cal/my_dist_pickle.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        img = cv2.undistort(img, mtx, dist, None, mtx)

def perspectiveTransform(image):
    imshape = image.shape
    height = imshape[0]
    width = imshape[1]

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

    increase_width_times = 1.6
    vertices = np.array([[
        [width*(0.5-mid_bottom_dist*increase_width_times), height*crop_from_bottom],
        [width*(0.5-mid_top_dist*increase_width_times), height*height_percentage],
        [width*(0.5+mid_top_dist*increase_width_times), height*height_percentage],
        [width*(0.5+mid_bottom_dist*increase_width_times), height*crop_from_bottom],
    ]], dtype=np.int32)

    image = tutil.region_of_interest(image, vertices)

    new_top_left=np.array([src[0,0], 0])
    new_top_right=np.array([src[3,0], 0])
    offset=[150, 0]
    
    dst = np.float32([
        src[0]+offset,
        new_top_left+offset,
        new_top_right-offset,
        src[3]-offset
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warp = cv2.warpPerspective(image, M, (width, height))
    return warp, Minv

video_gen()
#pipeline()