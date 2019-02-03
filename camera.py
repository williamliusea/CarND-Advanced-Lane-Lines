import matplotlib.image as mpimg
import glob
import numpy as np
import cv2
import pickle
import sys

class Camera():

    def load(self, filename):
        self.mtx, self.dist = pickle.load( open( filename, "rb" ) )

    def calibration(self, filename = None, nx = 9, ny = 6):
        images = glob.glob('camera_cal/*.jpg')
        objp = np.zeros((nx*ny,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

        objpoints = []
        imgpoints = []
        for name in images:
            img = mpimg.imread(name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        if (filename != None):
            pickle.dump((self.mtx, self.dist), open( filename, "wb"))

    def undistort(self, img):
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undist
