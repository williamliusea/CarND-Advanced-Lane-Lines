import matplotlib.image as mpimg
import os
import camera
import numpy as np
import cv2
import matplotlib.pyplot as plt
import config
import line
import time
import math

class ProcessImage():
    def __init__(self, config):
        self.config = config
        self.left_line = line.Line(config)
        self.right_line = line.Line(config)
        self.ploty = np.linspace(0, config.shape[0]-1, config.shape[0])
        self.perf= {'undistort':0,'binary':0,'warp':0,'find_lane':0,'fit_polynomial':0,'print':0}
        self.total = 0

    def color_binary(self, img):
        img = np.copy(img)
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        #h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]

        # taking h_channel into account, yellow's main band is
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= self.config.sx_thresh[0]) & (scaled_sobel <= self.config.sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.config.s_thresh[0]) & (s_channel <= self.config.s_thresh[1])] = 1
        # Stack each channel
        color_binary =  (sxbinary + s_binary) * 255
        return color_binary

    def warp_image(self, img):
        warped = cv2.warpPerspective(img, self.config.perspective_M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        return warped

    def find_lane_pixels(self, img):
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(img.shape[1]//2)
        if (self.left_line.bestx is None or self.left_line.detected == False):
            # Take a histogram of the bottom half of the image
            histogram_left = np.sum(img[img.shape[0]//2:,:img.shape[1]//2], axis=0)
            leftx_base = np.argmax(histogram_left)
            margin_left = self.config.margin_default
        else:
            leftx_base = self.left_line.bestx[-self.config.y_per_frame]
            # 4 is a heuristic number to narrow the search margin because we have confident on position based on previous value
            margin_left = self.config.margin_default/4

        if (self.right_line.bestx is None or self.right_line.detected == False):
            histogram_right = np.sum(img[img.shape[0]//2:,img.shape[1]//2:], axis=0)
            rightx_base = np.argmax(histogram_right) + midpoint
            margin_right = self.config.margin_default
        else:
            rightx_base = self.right_line.bestx[-self.config.y_per_frame]
            margin_right = self.config.margin_default/4

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(img.shape[0]//self.config.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        window_boundary = []

        # Step through the windows one by one
        for window in range(self.config.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            ### Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - margin_left
            win_xleft_high = leftx_current + margin_left
            win_xright_low = rightx_current - margin_right
            win_xright_high = rightx_current + margin_right
            window_boundary.append((win_xleft_low, win_xleft_high, win_xright_low, win_xright_high, win_y_low, win_y_high))

            ### Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzerox > win_xleft_low) & (nonzerox < win_xleft_high) &
                (nonzeroy > win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
            good_right_inds = ((nonzerox > win_xright_low) & (nonzerox < win_xright_high) &
                (nonzeroy > win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            ### The adaptive margin shall be related to a reasonable range of road curvature
            ### The idea is that a road shall not curve too much. Therefore, if it cannot find
            ### the line in one window, we can assume it will not be too far away from the
            ### the range of curvature in the next window.
            ### If you found > minpix pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position ###
            if len(good_left_inds) > self.config.minpix:
                new_leftx = int(np.mean(nonzerox[good_left_inds]))
                if (abs(new_leftx - leftx_current) < margin_left / 2):
                    leftx_current = new_leftx
                    margin_left = self.config.margin_default
                else:
                    margin_left = margin_left + self.config.margin_default
            else:
                margin_left = margin_left + self.config.margin_default

            if len(good_right_inds) > self.config.minpix:
                new_rightx = int(np.mean(nonzerox[good_right_inds]))
                if (abs(new_rightx - rightx_current) < margin_right / 2):
                    rightx_current = new_rightx
                    margin_right = self.config.margin_default
                else:
                    margin_right = margin_right + self.config.margin_default;
            else:
                margin_right = margin_right + self.config.margin_default;

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        self.left_line.allx = nonzerox[left_lane_inds]
        self.left_line.ally = nonzeroy[left_lane_inds]
        self.right_line.allx = nonzerox[right_lane_inds]
        self.right_line.ally = nonzeroy[right_lane_inds]

        return np.int32(window_boundary)

    def find_window_centroids(self, image):
        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(self.config.window_width) # Create our window template that we will use for convolutions
        window_height = np.int(image.shape[0]//self.config.nwindows)
        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
        l_convolution = np.convolve(window,l_sum)
        l_center = np.argmax(l_convolution)-self.config.window_width/2
        r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
        r_convolution = np.convolve(window,r_sum)
        r_center = np.argmax(r_convolution)-self.config.window_width/2+int(image.shape[1]/2)
        margin = self.config.margin_default
        # Add what we found for the first layer
        window_centroids.append((l_center,r_center))

        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(image.shape[0]/window_height)):
    	    # convolve the window into the vertical slice of the image
    	    image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
    	    conv_signal = np.convolve(window, image_layer)
    	    # Find the best left centroid by using past left center as a reference
    	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
    	    offset = self.config.window_width/2
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

    def printOverlay(self, undistort, warped):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_line.bestx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_line.bestx, self.ploty])))])
        # pts_left = np.array([np.transpose(np.vstack([self.left_line.recent_xfitted[-1], self.ploty]))])
        # pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_line.recent_xfitted[-1], self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        offset = (self.left_line.line_base_pos + self.right_line.line_base_pos)/2
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.config.perspective_Minv, (self.config.shape[1], self.config.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undistort, 1, newwarp, 0.3, 0)
        cv2.putText(result,'Radius of Curvature = ' + str((int)((self.left_line.radius_of_curvature + self.right_line.radius_of_curvature) / 2)) + "(m)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        placetext= "center"
        if offset > 0:
            placetext = "{0:.2f}".format(offset) +"m left of center"
        elif offset < 0:
            placetext = "{0:.2f}".format(-offset) +"m right of center"
        cv2.putText(result,'Vehicle is ' + placetext, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        return result

    def process_image_fit(self, undistort):
        warped = self.process_image_warped(undistort)
        start = time.time()
        self.find_lane_pixels(warped)
        self.perf['find_lane'] = self.perf['find_lane'] + (time.time() - start)
        out_img = np.dstack((warped, warped, warped))
        window_boundary = self.find_lane_pixels(warped)
        # Step through the windows one by one
        for window in window_boundary:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(window[0],window[4]),
            (window[1],window[5]),(0,255,0), 2)
            cv2.rectangle(out_img,(window[2],window[4]),
            (window[3],window[5]),(0,255,0), 2)
        start = time.time()
        self.left_line.fit_polynomial(self.ploty, warped.shape[1])
        self.right_line.fit_polynomial(self.ploty, warped.shape[1])
        self.perf['fit_polynomial'] = self.perf['fit_polynomial'] + (time.time() - start)
            ## Visualization ##
        # Colors in the left and right lane regions and Plots the left and right polynomials on the lane lines
        out_img[self.left_line.ally, self.left_line.allx] = [255, 0, 0]
        out_img[self.right_line.ally, self.right_line.allx] = [0, 0, 255]
        for j in range(len(self.left_line.recent_xfitted)):
            for i in range(len(self.ploty)):
                x = self.left_line.recent_xfitted[j][i];
                if (x >= 0 and x < undistort.shape[1]):
                    out_img[np.int32(self.ploty[i]), np.int32(x)] = [0, 255, 0]
        for j in range(len(self.right_line.recent_xfitted)):
            for i in range(len(self.ploty)):
                x = self.right_line.recent_xfitted[j][i];
                if (x >= 0 and x < undistort.shape[1]):
                    out_img[np.int32(self.ploty[i]), np.int32(x)] = [0, 255, 0]
        return out_img

    def process_image_warped(self, undistort):
        binary = self.process_image_binary(undistort)
        start = time.time()
        warped = self.warp_image(binary)
        self.perf['warp'] = self.perf['warp'] + (time.time() - start)
        return warped

    def process_image_binary(self, undistort):
        start = time.time()
        binary = self.color_binary(undistort)
        self.perf['binary'] = self.perf['binary'] + (time.time() - start)
        return binary

    def undistort(self, img):
        start = time.time()
        undistort = self.config.camera.undistort(img)
        self.perf['undistort'] = self.perf['undistort'] + (time.time() - start)
        return undistort

    def process_image(self, undistort):
        start = time.time()
        binary = self.color_binary(undistort)
        self.perf['binary'] = self.perf['binary'] + (time.time() - start)
        start = time.time()
        warped = self.warp_image(binary)
        self.perf['warp'] = self.perf['warp'] + (time.time() - start)
        start = time.time()
        self.find_lane_pixels(warped)
        self.perf['find_lane'] = self.perf['find_lane'] + (time.time() - start)
        start = time.time()
        self.left_line.fit_polynomial(self.ploty, warped.shape[1])
        self.right_line.fit_polynomial(self.ploty, warped.shape[1])
        self.perf['fit_polynomial'] = self.perf['fit_polynomial'] + (time.time() - start)
        start = time.time()
        output = self.printOverlay(undistort, warped)
        self.perf['print'] = self.perf['print'] + (time.time() - start)
        self.total = self.total + 1
        return output
