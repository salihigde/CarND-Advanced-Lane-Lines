import numpy as np

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #x values for detected line pixels
        self.left_fit = None  
        #y values for detected line pixels
        self.right_fit = None
        
        self.rightx = None
        self.righty = None
        self.leftx = None
        self.lefty = None
        self.left_fitx = None
        self.right_fitx = None
        self.ploty = None

        self.ym_per_pix = 30/720
        self.xm_per_pix = 3.7/700

        self.Minv = None

        self.warp_size = None
