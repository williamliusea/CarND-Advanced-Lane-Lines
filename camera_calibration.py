import camera
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
cam = camera.Camera()
cam.calibration("camera.p")
img = mpimg.imread('test_images/calibration1.jpg')
cam = camera.Camera()
cam.load("camera.p")
undist = cam.undistort(img)
mpimg.imsave('test_images_output/calibration1.jpg', undist)
