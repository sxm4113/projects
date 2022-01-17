
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:04:26 2020

Related paper :
    Adaptive Contrast Enhancement Using Local Region Stretching by S.Srinivasan.
    
@author: SYunMoon
"""

import numpy as np
from matplotlib import pyplot as plt
from skimage import io
import time

start = time.time()
img_read = io.imread('../../image/101ThermalTau2-S/thermal_027-S.png')

max_pix_val = np.max(img_read)
x_range =  np.arange(max_pix_val+1)

hist, bins = np.histogram(img_read, max_pix_val+1,[0,max_pix_val+1])

cdf = hist.cumsum()

###### dividing histogram into three regions #####
hist_1 = hist[0:85]
hist_2 = hist[85:170]
hist_3 = hist[170:max_pix_val+1]

cdf_1 = hist_1.cumsum()
cdf_2 = hist_2.cumsum()
cdf_3 = hist_3.cumsum()

fk_1 = hist_1/cdf_1[85-1]
fk_2 = hist_2/cdf_2[170-85-1]
fk_3 = hist_3/cdf_3[-1]

ffk_1 = fk_1.cumsum()
ffk_2 = fk_2.cumsum()
ffk_3 = fk_3.cumsum()

yout_1 = (0+(85-0) * ffk_1).astype('uint8')
yout_2 = (85+(170-85) * ffk_2).astype('uint8')
yout_3 = (170+(255-170) * ffk_3).astype('uint8')
yout = np.arange(max_pix_val+1)

yout[0:85] = yout_1
yout[85:170] = yout_2 
yout[170:256] = yout_3 
img2 = yout[img_read]

##### pseudo-variance #####
p_value_1 = np.arange(85)
p_time_h = p_value_1*hist_1
mean1 = np.sum(p_time_h)/cdf_1[85-1]
nu1 = np.sum((hist_1 * np.abs(p_value_1 - mean1)))/cdf_1[-1]

p_value_2 = np.arange(170-85)+85
b = p_value_2*hist_2
mean2 = np.sum(b)/cdf_2[170-85-1]
nu2 = np.sum((hist_2 * np.abs(p_value_2 - mean2)))/cdf_2[-1]

p_value_3 = np.arange(256-170)+170
b = p_value_3*hist_3
mean3 = np.sum(b)/cdf_3[-1]
nu3 = np.sum((hist_3 * np.abs(p_value_3 - mean3)))/cdf_3[-1]

new_image = np.zeros([img_read.shape[0],img_read.shape[1]])

for i in range(img_read.shape[0]):
    for j in range(img_read.shape[1]):
        
        if 0 <= img_read[i,j] <= 85:
            new_image [i,j] = (img2[i,j]+img_read[i,j])/2
        elif 85 < img_read[i,j] <= 170:
            new_image [i,j] = (img2[i,j]+img_read[i,j])/2
        elif 170 < img_read[i,j] <= 255:
            new_image [i,j] = (img2[i,j]+img_read[i,j])/2


plt.figure(1)
plt.imshow(img_read,'gray')
#plt.xlabel('Pixel Value')
#plt.ylabel('Number of Pixels')
plt.title('Image C')

plt.figure(2)
plt.imshow(img2,'gray')
plt.title('Processed')
end = time.time()

print("--- %s seconds ---" % (end - start))
#plt.plot(x_range ,img3, label= "stars", color= "green")
plt.show()