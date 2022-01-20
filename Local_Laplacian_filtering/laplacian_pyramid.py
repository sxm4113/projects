# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 08:53:07 2021

@author: SYunMoon
"""

import numpy as np 
import cv2
import time
 
start_time = time.process_time()

image = cv2.imread('C:/Users/SYunMoon/Python_program/image/input_images/barbara.png',0)
#lab = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
#whole_image, a, b = cv2.split(lab)  # split on 3 different channelsl, a, b = cv2.split(lab)  # split on 3 different channels
#whole_image = cv2.imread('C:/Users/SYunMoon/Downloads/input_images/input_images/input_png/barbara.png',0) 
#whole_image = np.resize(np.fromfile("C:/Users/SYunMoon/Desktop/airport_image/8.raw", dtype='uint8'),(480, 640))
#whole_image = np.resize(np.fromfile("C:/Users/SYunMoon/Desktop/raw_NU_corrected/second", dtype='uint16'),(1024, 1280))
  
#whole_image = (whole_image / np.max(whole_image))
   
whole_image =  (image) / 255.0
   
##sigma_r = 0.35, alpha = 0.25
sigma_r = 0.35
alpha = 0.25
beta = 1

width = len(whole_image[0])
height = len(whole_image)

# generate Gaussian pyramid
G = whole_image.copy()
gpA = [G]
for i in range(5):
    G = cv2.pyrDown(G)
    gpA.append(G)
    
#remap for detail
def fd (d):
    out = pow(d,alpha)
    if (alpha < 1):
       tau = smooth_step (0.01, 2 * 0.01, d* sigma_r )
       out = tau * out + (1-tau) * d
    return out

def smooth_step(xmin,xmax,x):
    y = (x - xmin)/(xmax - xmin)
    y = max(0,min(1,y))
    
    y = pow(y,2)* pow((y-2),2)
    return y

# =============================================================================
# remap_cal : calculate enhanced image(remapping) 
# input : partial image.  
#        pyramid[4] (32 * 32)   
#        pyramid[3] (64 x 64)   - use 189 x 189 partial image
#        pyramid[2] (128 x 128) - use 45 x 45 partial image 
#        pyramid[1] (256 x 256) - use 21 by 21 partial image
#        pyramid[0] (512 x 512) - use 9 by 9 partial image
# output : remapped image 
# =============================================================================
def remap_cal (image,g_0 ):
    
    p_width = len(image[0])
    p_height = len(image)
    
    remap = np.empty ((p_height,p_width))
    
    for y in range (0,p_height):
        for x in range (0,p_width):
            
            img[y,x] = min(max(img[y,x],g_0-sigma_r),g_0+sigma_r)
             
            delta = abs(image[y,x]-g_0)
   
            #if (delta < sigma_r):
            #    remap[y,x] = g_0 + np.sign(image[y,x] - g_0) * sigma_r * fd(delta/sigma_r)
            remap[y,x] = g_0 + np.sign(image[y,x] - g_0) * (beta * (delta - sigma_r) + sigma_r)
                
            #else:
            #    #remap for edge
            #    remap[y,x] = g_0 + np.sign(image[y,x] - g_0) * (beta * (delta - sigma_r) + sigma_r)
              
    return remap

#subtract upsampled image pyramid(l-1) from image pyramid(l). 
def lap_replace (image, pyramid):
    p_width = len(image[0])
    p_height = len(image)
    tempP = np.empty ((p_height, p_width))
    
    for y in range (0,p_height):
        for x in range (0,p_width):
            tempP[y,x] = image[y,x] - pyramid[y,x]
    
    return tempP

def diff (pyramidA, pyramidB):
    width = len(pyramidA[0])
    height = len(pyramidA)
    result = np.empty ((height, width))
    
    for i in range (height):
        for j in range (width):
            result[i,j] = pyramidA[i,j] - pyramidB[i,j]

    return result

def adding (pyramidA, pyramidB):
    width = len(pyramidB[0])
    height = len(pyramidB)
    result = np.empty ((height, width))
    
    for i in range (height):
        for j in range (width):
            result[i,j] = pyramidA[i,j] + pyramidB[i,j]

    return result

#find laplacian image pyramid
lpA = [gpA[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1],GE)
    
    #for odd number size
    #L = diff(gpA[i-1],GE)
    lpA.append(L)
    
# loop over laplacian image pyramid. 
# l = 0  :  512 x 512 laplacian pyramid 
for l in range (0,1):
    #use 9 x 9 sub region for the first laplacian pyramid
    #subregion_size = 3 * ((1 << (l + 2)) - 1); 
    subregion_size = 5
    subregion_r = int(subregion_size / 2)
    
    for y in range (0, int(height/(1+l))):
    #for y in range (0, 1):
        full_res_y = (1 << l) * y
        roi_y0 = full_res_y - subregion_r
        roi_y1 = full_res_y + subregion_r + 1
        row_range_start = max(0, roi_y0)
        full_res_roi_y = full_res_y - row_range_start
    
        for x in range (0 , int(width/(1+l))):
        #for x in range (0 , 1):
            full_res_x = (1 << l) * x
            roi_x0 = full_res_x - subregion_r
            roi_x1 = full_res_x + subregion_r + 1
            col_range_start = max(0, roi_x0)
            full_res_roi_x = full_res_x - col_range_start
                
             #g_0 is the reference value from the gaussian pyramid. 
            g_0 = gpA[l][y ,x] 
            
            img = whole_image[row_range_start: min(roi_y1, height),col_range_start: min(roi_x1, width)] 
  
            temp_pyramid = lap_replace (img, cv2.pyrUp(cv2.pyrDown(remap_cal (img,g_0))))
            
            #replace the first pyramid with remapped laplacian pyramid 
            #laplacian(x,y) is replaced by the center value of 9 x 9 laplacian pyramid 
            lpA[5-l][y,x] = temp_pyramid[full_res_roi_y,full_res_roi_x]

# now reconstruct
ls_ = lpA[0]
for i in range(1,6):
    ls_ = cv2.pyrUp(ls_) 
    ls_ = adding(ls_, lpA[i])

ls_[np.where(ls_ >1  )] = 1
ls_[np.where(ls_ <0  )] = 0

ls_ = ls_ * 255
ls_ = ls_.astype(np.uint8)


#********** color image **********
#lab = cv2.merge((ls_,a,b))  # merge channels
#img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
 
#cv2.imshow("color image", color_image)
#cv2.imshow("result", img2)
#**********************************//

cv2.imshow("input",color_image)
cv2.imshow("enhanced",np.uint8(ls_ * 255))
cv2.waitKey()
cv2.destroyAllWindows()

print("--- %s seconds ---" % (time.process_time() - start_time))

    
