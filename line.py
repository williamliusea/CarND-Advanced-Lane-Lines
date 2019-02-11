import numpy as np
import cv2
import math
import queue

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, config):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients based on meters for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in meters
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line,
        # positive value means on the right side of center
        # negative value means on the left side of center
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        # configuration of the detection
        self.config = config

    def fit_polynomial(self, ploty, width_pixel):
        # note that y is first and x is second because the plot is iterating on y, not x.
        new_fit = np.polyfit(self.ally * self.config.ym_per_pix, self.allx * self.config.xm_per_pix, 2)
        if (self.current_fit is not None):
            self.diffs = new_fit - self.current_fit
        self.current_fit = new_fit
        new_xfitted = (self.current_fit[0]*(ploty*self.config.ym_per_pix)**2 + self.current_fit[1]*ploty*self.config.ym_per_pix + self.current_fit[2])/self.config.xm_per_pix
        self.recent_xfitted.append(new_xfitted)
        self.recent_xfitted= self.recent_xfitted[len(self.recent_xfitted)-self.config.smooth_window:]
        self.bestx = np.mean(self.recent_xfitted, axis=0)
        self.best_fit = np.polyfit(ploty * self.config.ym_per_pix, self.bestx * self.config.xm_per_pix, 2)
        self.line_base_pos = (self.recent_xfitted[len(self.recent_xfitted) - 1][-1] - width_pixel/2) * self.config.xm_per_pix
        self.radius_of_curvature  = (1+(2*self.best_fit[0]*ploty[-1]*self.config.ym_per_pix + self.best_fit[1])**2)**1.5/2/np.absolute(self.best_fit[0])
