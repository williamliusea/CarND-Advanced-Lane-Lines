import camera
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
cam = camera.Camera()
cam.calibration("camera.p")
img = mpimg.imread('test_images/calibration2.jpg')
undist = cam.undistort(img)
mpimg.imsave('test_images_output/calibration2.jpg', undist)
