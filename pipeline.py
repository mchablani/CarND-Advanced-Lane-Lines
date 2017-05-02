# Define a class to receive the characteristics of each line detection
import cv2
import numpy as np

class Line():
    MAX_INVALID = 15
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        self.num_invalid = self.MAX_INVALID
        
        # previous fit
        self.left_fits = []
        self.right_fits = []
        
        # x values of the last 3 fits of the line
        self.recent_xfitted_left = [] 
        self.recent_xfitted_right = [] 
        
    def update_fit(self, left_fit, right_fit):
        self.left_fits.append(left_fit.copy())
        self.right_fits.append(right_fit.copy())
        if len(self.left_fits) > 10:
            self.left_fits.pop(0)
            self.right_fits.pop(0)
        
    def update_points(self, left_fitx, right_fitx):
        self.recent_xfitted_left.append(left_fitx.copy())
        self.recent_xfitted_right.append(right_fitx.copy())
        
        if len(self.recent_xfitted_left) > 10:
            self.recent_xfitted_left.pop(0)
            self.recent_xfitted_right.pop(0)
        
    def get_points(self):
        return self.recent_xfitted_left[-1], self.recent_xfitted_right[-1]

    def get_points_avg(self):
        weights=range(10,20,1)
        w = weights[:len(self.recent_xfitted_left)]
        # best_fit_left = np.average(self.recent_xfitted_left, axis=0, weights=w)
        # best_fit_right = np.average(self.recent_xfitted_right, axis=0, weights=w)
        best_fit_left = np.average(self.recent_xfitted_left, axis=0)
        best_fit_right = np.average(self.recent_xfitted_right, axis=0)
        return best_fit_left, best_fit_right

    def get_fits(self):
        return self.left_fits[-1], self.right_fits[-1]

    def get_fits_avg(self):
        weights=range(10,20,1)
        w = weights[:len(self.left_fits)]
        best_fit_left = np.average(self.left_fits, axis=0, weights=w)
        best_fit_right = np.average(self.right_fits, axis=0, weights=w)
        return best_fit_left, best_fit_right
    
    def reset(self):
        self.__init__()

def is_fit_within_range(prev_fit, fit, tolerance=0.2):
#    for i in range(0,len(fit)):
    for i in range(0,1):
        if abs((fit[i] - prev_fit[i])/fit[i]) > tolerance:
            return False
    return True    

def are_lines_valid(mid_point, left_fitx, right_fitx, left_fit, right_fit, validate_fit=True):
    valid = True
    if valid:
        left_line_base_pos = (left_fitx[-1] - mid_point)*3.7/700.0 # 3.7 meters is about 700 pixels in the x direction
        right_line_base_pos = (right_fitx[-1] - mid_point)*3.7/700.0 # 3.7 meters is about 700 pixels in the x direction
        maxdist = 5  # distance in meters for the lane
        mindist = 2
        if((abs(left_line_base_pos) > maxdist/2) or (abs(right_line_base_pos) > maxdist/2)):
            print('base lane too far away ', abs(left_line_base_pos), abs(right_line_base_pos))
            valid = False
        if((abs(left_line_base_pos) < mindist/2) or (abs(right_line_base_pos) < mindist/2)):
            print('base lane too close to center ', abs(left_line_base_pos), abs(right_line_base_pos))
            valid = False
    if valid:
        if (right_fitx - left_fitx).any() < 0:
            print('lines cross ', abs(left_line_base_pos), abs(right_line_base_pos))
            valid = False

    if valid and validate_fit:
        left_der = np.polyder(left_fit)
        right_der = np.polyder(right_fit)
        diff = abs(np.polyval(left_der, 720) - np.polyval(right_der, 720))
        if diff > 0.35:
            print('lines dont seem parallel', diff)
            print(left_fit)
            print(right_fit)
            print(left_der)
            print(right_der)
            valid = False

    return valid
    