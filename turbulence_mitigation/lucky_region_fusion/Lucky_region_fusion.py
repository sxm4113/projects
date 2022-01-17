# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 15:31:13 2021

@author: SYunMoon
"""

import numpy as np
import cv2  
import os

from skimage.filters import unsharp_mask

HEIGHT = 512
WIDTH = 640

in_dir = "D:/image/turbulence_midigation/"

# =============================================================================
# IQM : find image quality map, the magnitue of image gradient.
#
# parameters  
#        image : image in uint8.
# return
#        image quality map
# =============================================================================
def IQM(image):

    sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)

    magnitude = pow(sobelx,2) + pow(sobely,2) 
    intensity = sum(sum(image))

    #normalize
    J = magnitude / intensity
    IQM_map = cv2.blur(J,(17,17))
    
    return IQM_map

# =============================================================================
# IQM : find image quality map, the magnitue of image gradient.
#
# parameters
#        new_image : image in uint8.
#        Msynthetic : previously generated synthetic image
# return
#        new synthetic image
# =============================================================================
def append_image (new_image, Msynthetic):

    new_Msynthetic = np.zeros_like(new_image)
    delta = np.zeros((HEIGHT, WIDTH))

    M = IQM(new_image)
    MS = IQM(Msynthetic)

    index = np.where(M > MS)
    # print ("replaced pixels = ", len(index[0]))
    delta[index] = (M - MS)[index]

    if np.max(delta) != 0:
        delta = delta / np.max(delta)

        new_Msynthetic = np.uint8((1 - delta) * Msynthetic + delta * new_image)

    return data, new_Msynthetic

# =============================================================================
# Main function
# =============================================================================
data = []
processed = []

for count, i in enumerate(os.listdir(in_dir)):

    new_image = cv2.imread(in_dir + i,0)
    new_image = cv2.resize(new_image,(640,512),interpolation = cv2.INTER_AREA)
    
    if count < 50:
        data.append(new_image)

    #first synthetic image is average of first 50 frames
    if count == 50:
        Msynthetic = np.mean(data,axis=0)
  
    elif count >50:
        data, Msynthetic= append_image(new_image, Msynthetic)
    
        processed.append(Msynthetic)
    if count ==55:
        break

    print ("count = " , count )

