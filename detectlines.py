import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from Line import *

class DetectLines:
    def __init__(self, orginal_img, warped_img, params):        
        self.orginal_img = orginal_img
        self.warped_img = warped_img
        self.params = params

    def process_line_detection(self):
        out_img = self.fit_polynomial(self.warped_img)

        imshape = out_img.shape
        height = imshape[0]
        width = imshape[1]

        warp = cv2.warpPerspective(out_img, self.params.Minv, (width, height))
        self.params.warp_size = warp.shape
        
        result = cv2.addWeighted(self.orginal_img, 1.0, warp, 1.0, 0.0)

        return result

    def add_radius_and_distance_to_img(self, result):
        width = result.shape[1]

        xm_per_pix = 3.7/width
        camera_center = (self.params.left_fitx[-1] + self.params.right_fitx[-1])/2
        center_diff = round(((camera_center - width/2) * xm_per_pix) * 1000, 1)
        car_pos = 'left'
        if(center_diff <= 0):
            car_pos = 'right'

        left_curverad, right_curverad = self.measure_curvature_real(self.params.warp_size)

        curverad = round((round(right_curverad,1) + round(left_curverad,1))/2, 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, 'Radius of Curvature (m): ' + str(curverad), (50,50), font, 1, (0,0,255), 2)
        cv2.putText(result, 'Distance from center (cm): ' + str(center_diff) + ' in ' + car_pos, (50, 100), font, 1, (0,0,255), 2)

        return result

    def find_lane_pixels(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        margin = 100
        minpix = 50

        window_height = np.int(binary_warped.shape[0]//nwindows)

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            pass

        self.params.leftx = nonzerox[left_lane_inds]
        self.params.lefty = nonzeroy[left_lane_inds] 
        self.params.rightx = nonzerox[right_lane_inds]
        self.params.righty = nonzeroy[right_lane_inds]

        return out_img

    def fit_polynomial(self, binary_warped):
        out_img = self.find_lane_pixels(binary_warped)

        self.params.left_fit = np.polyfit(self.params.lefty, self.params.leftx, 2)
        self.params.right_fit = np.polyfit(self.params.righty, self.params.rightx, 2)

        self.params.ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        try:
            self.params.left_fitx = self.params.left_fit[0]*self.params.ploty**2 + self.params.left_fit[1]*self.params.ploty + self.params.left_fit[2]
            self.params.right_fitx = self.params.right_fit[0]*self.params.ploty**2 + self.params.right_fit[1]*self.params.ploty + self.params.right_fit[2]
        except TypeError:
            print('The function failed to fit a line!')
            self.params.left_fitx = 1*self.params.ploty**2 + 1*self.params.ploty
            self.params.right_fitx = 1*self.params.ploty**2 + 1*self.params.ploty

        out_img[self.params.lefty, self.params.leftx] = [255, 0, 0]
        out_img[self.params.righty, self.params.rightx] = [0, 0, 255]

        return self.paint_road_to_green(out_img)
    
    def paint_road_to_green(self, out_img):
        pts_left = np.array([np.transpose(np.vstack([self.params.left_fitx, self.params.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.params.right_fitx, self.params.ploty])))])
        pts = np.hstack((pts_left, pts_right))        
        cv2.fillPoly(out_img, np.int_([pts]), (0,240, 0))
        return out_img

    def write_curvature_to_image(distance_from_center, r_left, r_right):
        font = cv2.FONT_HERSHEY_SIMPLEX
        str1 = str('distance from center: '+str(distance_from_center)+'cm')
        cv2.putText(result,str1,(430,630), font, 1,(0,0,255),2,cv2.LINE_AA)
        if r_left and r_right:
            curvature = 0.5*(round(r_right/1000,1) + round(r_left/1000,1))
            str2 = str('radius of curvature: '+str(curvature)+'km')
            cv2.putText(result,str2,(430,670), font, 1,(0,0,255),2,cv2.LINE_AA)    
           
    def measure_curvature_real(self, warped_size):
        left_fit_cr = np.polyfit(self.params.lefty*self.params.ym_per_pix, self.params.leftx*self.params.xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.params.righty*self.params.ym_per_pix, self.params.rightx*self.params.xm_per_pix, 2)

        y_eval = np.max(warped_size[0])
        
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*self.params.ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*self.params.ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        return round(left_curverad, 3), round(right_curverad, 3)