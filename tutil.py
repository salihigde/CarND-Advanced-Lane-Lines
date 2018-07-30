import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l = hls[:,:,1]
    if(orient=='x'):
        sobel = cv2.Sobel(l, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(l, cv2.CV_64F, 0, 1)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def color_thresh(image, s_thresh=(0,255), v_thresh=(0,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    l = hls[:,:,1]
    l_binary = np.zeros_like(l)
    l_binary[(l >= v_thresh[0]) & (l <= v_thresh[1])] = 1

    s = hls[:,:,2]
    s_binary = np.zeros_like(s)
    s_binary[(s >= s_thresh[0]) & (s <= s_thresh[1])] = 1
    
    combined = np.zeros_like(s)
    combined[(s_binary == 1) & (l_binary == 1)] = 1
    
    return combined
