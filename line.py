import numpy as np
import cv2
import math
import queue

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, config):
        # was the line detected in the last iteration?
        self.detected = False
        # the last n Fits
        self.recent_fit = []
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
        if (self.allx is None or len(self.allx) < 20):
            self.detected = False
            pass
        self.detected = True
        # note that y is first and x is second because the plot is iterating on y, not x.
        new_fit = np.polyfit(self.ally * self.config.ym_per_pix, self.allx * self.config.xm_per_pix, 2)
        if (self.current_fit is not None):
            self.diffs = new_fit - self.current_fit

        self.current_fit = new_fit
        new_xfitted = (self.current_fit[0]*(ploty*self.config.ym_per_pix)**2 + self.current_fit[1]*ploty*self.config.ym_per_pix + self.current_fit[2])/self.config.xm_per_pix
        std=0
        if (len(self.recent_xfitted) >= self.config.smooth_window):
            # use the previous best fit to project the previous fits in the new frame
            for i in range(len(self.recent_xfitted)):
                # moving the line x=a(y-h)^2+b(y-h)+c+k, h=y0-y1, k=x0-x1,
                xfitted = self.recent_xfitted[i]
                h=(ploty[-1]-ploty[-self.config.y_per_frame])*self.config.ym_per_pix
                k=(xfitted[-1]-xfitted[-self.config.y_per_frame])*self.config.xm_per_pix
                fit = self.recent_fit[i]
                xfitted = (fit[0]*((ploty+h)*self.config.ym_per_pix)**2 + fit[1]*(ploty+h)*self.config.ym_per_pix + fit[2]+k)/self.config.xm_per_pix
                self.recent_xfitted[i] = xfitted
            avg_xfitted = np.average(self.recent_xfitted, axis=0)
            for i in range(len(new_xfitted)):
                std=std+(avg_xfitted[i]-new_xfitted[i])**2
            std = math.sqrt(std/len(new_xfitted))
        if (std < 100):
            self.recent_xfitted.append(new_xfitted)
            if (len(self.recent_xfitted)> self.config.smooth_window):
                self.recent_xfitted= self.recent_xfitted[1:]
            # speed up the adaption to the new fit by only average the recent 3
            self.bestx = np.average(self.recent_xfitted[-3:], axis=0)
            self.best_fit = np.polyfit(ploty * self.config.ym_per_pix, self.bestx * self.config.xm_per_pix, 2)
            self.recent_fit.append(self.best_fit)
            if (len(self.recent_fit)> self.config.smooth_window):
                self.recent_fit= self.recent_fit[1 :]
        self.line_base_pos = (self.recent_xfitted[len(self.recent_xfitted) - 1][-1] - width_pixel/2) * self.config.xm_per_pix
        self.radius_of_curvature  = (1+(2*self.best_fit[0]*ploty[-1]*self.config.ym_per_pix + self.best_fit[1])**2)**1.5/2/np.absolute(self.best_fit[0])
