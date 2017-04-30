# Define a class to receive the characteristics of each line detection
import cv2
import numpy as np

class Line():
    MAX_INVALID = 5
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        self.num_invalid = self.MAX_INVALID
        
        # previous fit
        self.left_fit = []
        self.right_fit = []
        
        # x values of the last 3 fits of the line
        self.recent_xfitted_left = [] 
        self.recent_xfitted_right = [] 
        
    def update_fit(self, left_fit, right_fit):
        self.left_fit = left_fit
        self.right_fit = right_fit
#         self.recent_fits.append([left_fit, right_fit])
#         if len(self.recent_fits) > 3:
#             self.recent_fits.pop(0)
        
    def update_points(self, left_fitx, right_fitx):
        self.recent_xfitted_left.append(left_fitx.copy())
        self.recent_xfitted_right.append(right_fitx.copy())
        
        if len(self.recent_xfitted_left) > 3:
            self.recent_xfitted_left.pop(0)
            self.recent_xfitted_right.pop(0)
        
    def get_points(self):
        return self.recent_xfitted_left[-1], self.recent_xfitted_right[-1]

    def get_points_avg(self):
        weights=range(10,20,2)
        w = weights[:len(self.recent_xfitted_left)]
        best_fit_left = np.average(self.recent_xfitted_left, axis=0, weights=w)
        best_fit_right = np.average(self.recent_xfitted_right, axis=0, weights=w)
        return best_fit_left, best_fit_right

    def reset(self):
        self.__init__()
        
def are_lines_valid(mid_point, left_fitx, right_fitx):
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
#         if valid:
#             left_line_avg_pos = (np.average(left_fitx) - mid_point)*3.7/700.0 # 3.7 meters is about 700 pixels in the x direction
#             right_line_avg_pos = (np.average(right_fitx) - mid_point)*3.7/700.0 # 3.7 meters is about 700 pixels in the x direction
#             maxdist = 5  # distance in meters for the lane
#             if((abs(left_line_avg_pos) > maxdist/2) or (abs(right_line_avg_pos) > maxdist/2)):
#                 print('avg lane too far away ', abs(left_line_avg_pos), abs(right_line_avg_pos))
#                 valid = False          return True
    return valid
