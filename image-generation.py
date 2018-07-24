import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

images = glob.glob('test_images/test*.jpg')

def undistort():
    dist_pickle = pickle.load( open( "camera_cal/my_dist_pickle.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        img = cv2.undistort(img, mtx, dist, None, mtx)

        undistorted_img_name = './test_images/undistorted' + str(i) + '.jpg'
        cv2.imwrite(undistorted_img_name, img)