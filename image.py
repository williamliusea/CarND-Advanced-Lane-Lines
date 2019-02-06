import argparse
import matplotlib.image as mpimg
import os
import camera
import numpy as np
import cv2
import matplotlib.pyplot as plt
import processimage
import config
import line

def find_lane_pixels(binary_warped, left_line, right_line):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    window_boundary = processor.find_lane_pixels(binary_warped, left_line, right_line)
    # Step through the windows one by one
    for window in window_boundary:
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(window[0],window[4]),
        (window[1],window[5]),(0,255,0), 2)
        cv2.rectangle(out_img,(window[2],window[4]),
        (window[3],window[5]),(0,255,0), 2)

    return out_img

# Fits a curve using actual world dimension in meters
def fit_polynomial(binary_warped, left_line, right_line):
    # Find our lane pixels first
    out_img = find_lane_pixels(binary_warped, left_line, right_line)
    left_line.fit_polynomial(ploty, binary_warped.shape[1])
    right_line.fit_polynomial(ploty, binary_warped.shape[1])
    ## Visualization ##
    # Colors in the left and right lane regions and Plots the left and right polynomials on the lane lines
    out_img[left_line.ally, left_line.allx] = [255, 0, 0]
    out_img[right_line.ally, right_line.allx] = [0, 0, 255]
    out_img[np.int32(ploty), np.int32(left_line.recent_xfitted)] = [0, 255, 0]
    out_img[np.int32(ploty), np.int32(right_line.recent_xfitted)] = [0, 255, 0]
    return out_img

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-
    (level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):
    min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_print_window_centroids(warped):
    window_height = np.int(warped.shape[0]//config.nwindows)
    window_centroids = processor.find_window_centroids(warped)
    # If we found any window centers
    if len(window_centroids) > 0:
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
    	    l_mask = window_mask(config.window_width,window_height,warped,window_centroids[level][0],level)
    	    r_mask = window_mask(config.window_width,window_height,warped,window_centroids[level][1],level)
    	    # Add graphic points from window mask here to total pixels found
    	    l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
    	    r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage= np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)
    return output

def printOverlay(undistort, warped, left_line, right_line, src, dest):
    Minv = cv2.getPerspectiveTransform(dest, src)# np.linalg.inv(cv2.getPerspectiveTransform(src, dest))
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_line.recent_xfitted, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.recent_xfitted, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    offset = (left_line.line_base_pos + right_line.line_base_pos)/2
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistort, 1, newwarp, 0.3, 0)
    cv2.putText(result,'Radius of Curvature = ' + str((int)((left_line.radius_of_curvature + right_line.radius_of_curvature) / 2)) + "(m)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    placetext= "center"
    if offset > 0:
        placetext = "{0:.2f}".format(offset) +"m left of center"
    elif offset < 0:
        placetext = "{0:.2f}".format(-offset) +"m right of center"
    cv2.putText(result,'Vehicle is ' + placetext, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return result

parser = argparse.ArgumentParser(description='Process single image. Or all image in test_images/. \nLoad from test_images/ and output to test_images_output/.')
parser.add_argument('filename', type=str, nargs='?', default='',
                   help='filename in test_images directory')
args = parser.parse_args()
if args.filename == '':
    filenames = os.listdir("test_images/")
else:
    filenames = [args.filename]
# src= np.float32([[564, 474], [717,474], [246, 700], [1060, 700]])
# dst= np.float32([[246, 474], [1060,474], [246, 700], [1060, 700]])
cam = camera.Camera()
cam.load('camera.p')
left_line = line.Line()
right_line = line.Line()
processor = processimage.ProcessImage(config)
for name in filenames:
    #reading in an image
    image = mpimg.imread('test_images/'+name)
    img_size = (image.shape[1], image.shape[0])
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    src = np.float32(
        [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
        [((img_size[0] / 6) - 25), img_size[1]],
        [(img_size[0] * 5 / 6) + 35, img_size[1]],
        [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])
    config = config.Config()
    config.setPerspectiveMatrix(src, dst)
    processor.config = config
    left_line.config = config
    right_line.config = config
    undistort = cam.undistort(image)
    mpimg.imsave('test_images_output/undistort_'+name, undistort)
    binary = processor.color_binary(undistort)
    mpimg.imsave('test_images_output/binary_'+name, binary)
    warped = processor.warp_image(binary)
    mpimg.imsave('test_images_output/warped_'+name, warped)
    centroids = find_print_window_centroids(warped)
    mpimg.imsave('test_images_output/centroids'+name, centroids)
    fitted = fit_polynomial(warped, left_line, right_line)
    mpimg.imsave('test_images_output/fitted_'+name, fitted)
    result = printOverlay(undistort, warped, left_line, right_line, src, dst)
    mpimg.imsave('test_images_output/overlay_'+name, result)
