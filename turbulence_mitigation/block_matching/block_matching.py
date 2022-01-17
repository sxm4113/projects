# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 09:41:27 2021

@author: SYunMoon
"""

import numpy as np
import cv2
import os
from skimage.filters import unsharp_mask
import time
import matplotlib.pyplot as plt

# =============================================================================
# display_image : Display images.
#
# parameters
#        image1 : 1st image.
#        image2 : 2nd image.
# =============================================================================
def display_image(image1,image2=0):
    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(np.uint8(image1),'gray')
    if len(image2):
        plt.subplot(1,2,2)
        plt.axis('off')
        plt.imshow(np.uint8(image2),'gray')


# =============================================================================
# Block Matching Algorithm : this function calculates motion vector for each pixel.
#
# parameters
#        image : single image needs to be registgered.
#        reference : reference image.
# return
#        registered image to the reference.
# =============================================================================
def BMA_image(image, reference):

    subregion_size = 21
    window_size = 40

    subregion_r = subregion_size // 2
    window_r = window_size // 2
    map_x = np.zeros((HEIGHT, WIDTH))
    map_y = np.zeros((HEIGHT, WIDTH))

    for y in range(0, HEIGHT):
        roi_y0 = y - subregion_r
        roi_y1 = y + subregion_r + 1

        #prevent negative start point (y) for subregion
        if roi_y0 < 0:
            row_range_start = 0
        else:
            row_range_start = roi_y0

        w_roi_y0 = y - window_r
        w_roi_y1 = y + window_r + 1

        # prevent negative start point (y)for search window
        if w_roi_y0 < 0:
            w_row_range_start = 0
        else:
            w_row_range_start = w_roi_y0

        for x in range(0, WIDTH):
            roi_x0 = x - subregion_r
            roi_x1 = x + subregion_r + 1

            # prevent negative start point (x) for subregion
            if roi_x0 < 0:
                col_range_start = 0
            else:
                col_range_start = roi_x0

            w_roi_x0 = x - window_r
            w_roi_x1 = x + window_r + 1

            # prevent negative start point (x)for search window
            if w_roi_x0 < 0:
                w_col_range_start = 0
            else:
                w_col_range_start = w_roi_x0

            #define search window
            window = reference[w_row_range_start: min(w_roi_y1, HEIGHT),
                     w_col_range_start: min(w_roi_x1, WIDTH)].astype(np.float32)


            img = image[row_range_start: min(roi_y1, HEIGHT),
                  col_range_start: min(roi_x1, WIDTH)].astype(np.float32)

            #find maching point within the search window
            blockM = cv2.matchTemplate(window, img, cv2.TM_SQDIFF)
            index = np.where(blockM == np.min(blockM))

            index_x = max(0, x - window_r)
            index_y = max(0, y - window_r)

            map_y[y, x] = index[0][0] + index_y
            map_x[y, x] = index[1][0] + index_x

    return cv2.remap(reference, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)

# =============================================================================
# append_image : calculate average of 30 frames while it keeps latest 30 frames in the list.
#
# parameters
#        data : latest list of image.
#        average : latest average of 30 frames.
#        new_image : a new image to add
# return
#        data : updated image list.
#        new_average : average after new image added.
# =============================================================================
def append_image (data, average, new_image):

    data.pop(0)
    data.append(new_image)
    new_average = np.mean(data,axis=0)

    return data, new_average

# =============================================================================
# convert_avi : generate avi file from processed frames.
#
# parameters
#        processed : list of processed images.
# return
#        video file
# =============================================================================
def convert_avi(processed):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('block_matching.avi', fourcc, 60, (WIDTH, HEIGHT), 0)

    for i in range(len(processed)):
        # image = cv2.imread(collection+filename)
        out.write(processed[i])
    out.release()

# =============================================================================
# edge_info : detece edge using canny detection
#
# parameters
#        edge : Region of interest
# return
#        x : x coordinates of edge
#        y : y coordinates of edge
#        edge detection : detected edge
# =============================================================================
def edge_info(edge):

    edge_detection = cv2.Canny(image=np.uint8(edge), threshold1=100, threshold2=200)  # Canny Edge

    index = np.where(edge_detection == 255)

    x = index[0]
    y = index[1]

    return x,y,edge_detection

# =============================================================================
# linearity_cal : calculates linearity of edge.
#                 The linearity of a line can be used to check the performance of the algorithm
#                 because the line in the image should be straight after turbulence mitigation.
#
# parameters
#        image : input image
# return
#        it shows 1. block matched image
#                 2. edge
#                 3. detected edge
#                 4. linear regression
# =============================================================================
def linearity_cal (image):

    rough_edge = image[100:200, 390:430].copy()
    rough_x,rough_y,_ = edge_info(rough_edge)

    x_start = 390 + np.mean(rough_y,dtype=int) - 15
    x_end = 390 + np.mean(rough_y,dtype=int) + 5
    edge = image[100:200, x_start:x_end].copy()

    x,y,edge_detection = edge_info(edge)

    k, d = np.polyfit(x, y, 1)
    y_pred = k * x + d

    image_box = cv2.rectangle(image, (x_start, 100), (x_end, 200), (255, 255, 255), 2).copy()

    fig = plt.figure(figsize = (8,8))

    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(image_box, 'gray')
    ax1.set_title('BMA')
    ax1.axis('off')

    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(edge,'gray')
    ax2.set_title ('Edge')
    ax2.axis('off')

    ax3 = fig.add_subplot(2,2,3)
    ax3.imshow(edge_detection,'gray')
    ax3.set_title('Edge detection')
    ax3.axis('off')

    ax4 = fig.add_subplot(2,2,4)
    ax4.plot(x, y, label='Detected edge')
    ax4.plot(x, y_pred, label='Linear Trend Line')
    ax4.legend(loc='upper right')
    ax4.set_title('Linear regression')
    ax4.set_ylim([max(0,int(np.mean(y))-20), int(np.mean(y))+20])

    plt.savefig('result.png')
    plt.show()
    residuals = y - y_pred
    SSR = np.sum(residuals ** 2)
    print("SSR:{}".format(SSR))


# =============================================================================
# Main function 
# =============================================================================

###  create empty list to store images.
processed = []
data = []

WIDTH = 640
HEIGHT = 512

in_dir = 'D:/image/turbulence_midigation'

for count, i in enumerate( os.listdir(in_dir)):

    new_image = cv2.imread(os.path.join(in_dir,i),0) 
    new_image = cv2.resize(new_image,(WIDTH,HEIGHT),interpolation = cv2.INTER_AREA)

    #first 30 frames are used to calculate reference.
    if count>= 0 and count <= 30:
        data.append(new_image)
        if count == 30:
            average = np.mean(data,axis=0)

    elif count >31 and count <= 33:
        data, average_ = append_image(data, average, new_image)
        processed.append(BMA_image(new_image,average))

    elif count > 33:
        break
    print ("count = ", count)

convert_avi(processed)
#linearity_cal(processed[0])

 
 
