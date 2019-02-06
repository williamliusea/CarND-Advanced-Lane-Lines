import cv2

class Config():
    def __init__(self):
        self.s_thresh = (200, 255)
        self.sx_thresh = (20, 100)
        self.margin_default = 100
        self.minpix = 100
        self.nwindows = 9
        self.xm_per_pix = 3.7 / 629
        self.ym_per_pix = 30 / 720
        self.window_width = 50

    def setPerspectiveMatrix(self, src, dest):
        self.perspective_M = cv2.getPerspectiveTransform(src, dest)
