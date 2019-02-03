import argparse
import matplotlib.image as mpimg
import os
import camera
import numpy as np
import cv2
import matplotlib.pyplot as plt

def color_binary(img, s_thresh=(200, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary =  (sxbinary + s_binary) * 255
    return color_binary

def warp_image(img, src, dest):
    M = cv2.getPerspectiveTransform(src, dest)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    #print(len(nonzero), '\n', len(nonzerox), '\n', len(nonzeroy))
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2)

        ### Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzerox > win_xleft_low) & (nonzerox < win_xleft_high) &
            (nonzeroy > win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
        #print(((nonzerox > win_xleft_low) & (nonzerox < win_xleft_high) &
        #    (nonzeroy > win_y_low) & (nonzeroy < win_y_high)).nonzero())
        good_right_inds = ((nonzerox > win_xright_low) & (nonzerox < win_xright_high) &
            (nonzeroy > win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        ### If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

# Fits a curve using actual world dimension in meters
def fit_polynomial(binary_warped, xm_per_pix, ym_per_pix):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # note that y is first and x is second because the plot is iterating on y, not x.
    left_fit = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit = np.polyfit(righty* ym_per_pix, rightx * xm_per_pix, 2)
    # # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = (left_fit[0]*(ploty*ym_per_pix)**2 + left_fit[1]*ploty*ym_per_pix + left_fit[2])/xm_per_pix
        right_fitx = (right_fit[0]*(ploty*ym_per_pix)**2 + right_fit[1]*ploty*ym_per_pix + right_fit[2])/xm_per_pix
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    ## Visualization ##
    # Colors in the left and right lane regions and Plots the left and right polynomials on the lane lines
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    out_img[np.int32(ploty), np.int32(left_fitx)] = [0, 255, 0]
    out_img[np.int32(ploty), np.int32(right_fitx)] = [0, 255, 0]
    return left_fit, right_fit, out_img

def find_window_centroids(image, window_width, window_height, margin):
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)

    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
	    # convolve the window into the vertical slice of the image
	    image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
	    conv_signal = np.convolve(window, image_layer)
	    # Find the best left centroid by using past left center as a reference
	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
	    offset = window_width/2
	    l_min_index = int(max(l_center+offset-margin,0))
	    l_max_index = int(min(l_center+offset+margin,image.shape[1]))
	    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
	    # Find the best right centroid by using past right center as a reference
	    r_min_index = int(max(r_center+offset-margin,0))
	    r_max_index = int(min(r_center+offset+margin,image.shape[1]))
	    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
	    # Add what we found for that layer
	    window_centroids.append((l_center,r_center))

    return window_centroids

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_print_window_centroids(warped):
    window_width = 50
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching
    window_centroids = find_window_centroids(warped, window_width, window_height, margin)
    # If we found any window centers
    if len(window_centroids) > 0:
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
    	    l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
    	    r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
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

def measure_curvature_pixels(left_fit, right_fit, y_eval):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    # y_eval = np.max(ploty)

    left_curverad = (1+(2*left_fit[0]*y_eval + left_fit[1])**2)**1.5/2/np.absolute(left_fit[0])
    right_curverad = (1+(2*right_fit[0]*y_eval + right_fit[1])**2)**1.5/2/np.absolute(right_fit[0])

    return left_curverad, right_curverad

def printOverlay(undistort, warped, left_fit, right_fit, src, dest, curvature):
    Minv = cv2.getPerspectiveTransform(dest, src)# np.linalg.inv(cv2.getPerspectiveTransform(src, dest))
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Generate x and y values for plotting

    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    try:
        left_fitx = (left_fit[0]*(ploty*ym_per_pix)**2 + left_fit[1]*(ploty*ym_per_pix) + left_fit[2])/xm_per_pix
        right_fitx = (right_fit[0]*(ploty*ym_per_pix)**2 + right_fit[1]*(ploty*ym_per_pix) + right_fit[2])/xm_per_pix
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    y = warped.shape[0]-1;
    offset = ((right_fit[0]*(y*ym_per_pix)**2 + right_fit[1]*(y*ym_per_pix) + right_fit[2] + left_fit[0]*(y*ym_per_pix)**2 + left_fit[1]*(y*ym_per_pix) + left_fit[2]) - warped.shape[1]  * xm_per_pix)/2
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistort, 1, newwarp, 0.3, 0)
    cv2.putText(result,'Radius of Curvature = ' + str((int)(curvature)) + "(m)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
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
for name in filenames:
    #reading in an image
    image = mpimg.imread('test_images/'+name)
    img_size = (image.shape[1], image.shape[0])
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/629 # meters per pixel in x dimension
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
    undistort = cam.undistort(image)
    mpimg.imsave('test_images_output/undistort_'+name, undistort)
    binary = color_binary(undistort)
    mpimg.imsave('test_images_output/binary_'+name, binary)
    warped = warp_image(binary, src, dst)
    mpimg.imsave('test_images_output/warped_'+name, warped)
    centroids = find_print_window_centroids(warped)
    mpimg.imsave('test_images_output/centroids'+name, centroids)
    left_fit, right_fit, fitted = fit_polynomial(warped, xm_per_pix, ym_per_pix)
    left_curverad, right_curverad = measure_curvature_pixels(left_fit, right_fit, img_size[1]*ym_per_pix)
    mpimg.imsave('test_images_output/fitted_'+name, fitted)
    result = printOverlay(undistort, warped, left_fit, right_fit, src, dst, (left_curverad + right_curverad) / 2)
    mpimg.imsave('test_images_output/overlay_'+name, result)
