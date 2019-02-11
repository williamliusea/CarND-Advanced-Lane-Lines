import cv2
import math

class Config():
    def __init__(self):
        self.s_thresh = (180, 255)
        self.sx_thresh = (20, 100)
        # self.margin_default = 100
        self.minpix = 100
        self.nwindows = 9
        self.xm_per_pix = 3.7 / 629
        self.ym_per_pix = 30 / 720 # this is based on the assumption that the road is flat. No going up hill or downhill
        self.min_r = 200 # minimal radius of curvature is 200m
        self.margin_default = (-math.sqrt(self.min_r**2-(720*self.ym_per_pix)**2) + 200)/self.xm_per_pix/self.nwindows
        print(self.margin_default)
        self.window_width = 50
        self.shape = []
        self.smooth_window = 12

    def setPerspectiveMatrix(self, src, dest):
        self.perspective_M = cv2.getPerspectiveTransform(src, dest)
        self.perspective_Minv = cv2.getPerspectiveTransform(dest, src)
