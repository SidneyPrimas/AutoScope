"""
    File name: preprocess.py
    Author: Sidney Primas
    Date created: 05/02/2017
    Python Version: 2.7
    Description: Performs histogram equalization on images in order to better visualize particles. 
    This is purely for the purpose of making the image easier to visualize. It doesn't help with most processing techniques (like watershed)
"""


# Base packages 
import numpy as np
import matplotlib.pyplot as plt


# Import OpenCV
import cv2

# Reversed Lens
target_file = "./data/20170425/reversed_lens/10um/1.bmp"
target_file = "./data/20170425/reversed_lens/baf3/t_2.bmp"
# Microscope
#target_file = "./data/20170425/microscope/10um/10um_final/Image1.tif"
#target_file = "./data/20170425/microscope/BAF3/baf3_final/Image1.tif"

im_original = cv2.imread(target_file, 0)

# Make sure input figure is in an 8-bit format.
im_original = cv2.convertScaleAbs(im_original, alpha = (255.0/np.amax(im_original)))

# Peform histogram equalization. 
im_equalized = cv2.equalizeHist(im_original)

# Contrast Limited Adaptive Histogram Equalization
clahe = cv2.createCLAHE(clipLimit=50.0, tileGridSize=(20,20))
im_clahe = clahe.apply(im_original)



## Interpolation for imshow: By default, we interpolate the base pixels in the image when we use image show. Set to 'nearest' to run off interpolation.
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.title("Original: Image")
imgplot = plt.imshow(im_original, cmap='gray', interpolation='nearest')
plt.axis('off')
# Plot histogram
## normed indicates that the integral of the histogram is 1
## flatten(): Flattens the image into a single array
fig.add_subplot(1,2,2)
plt.title("Histogram of Pixels")
plt.xlabel("Pixel Intensity")
hist, _, _ = plt.hist(im_original.flatten(), 256, normed=1, facecolor='green')
# Get the cumulative sum of the histogram.
cdf = hist.cumsum()
# Normalize so that the CDF is represented on same scale as histogram
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.plot(cdf_normalized, color = 'b')


fig = plt.figure()
fig.add_subplot(1,2,1)
plt.title("Histogram Equalization")
imgplot = plt.imshow(im_equalized, cmap='gray', interpolation='nearest')
plt.axis('off')
# Plot histogram
fig.add_subplot(1,2,2)
plt.title("Histogram of Pixels")
plt.xlabel("Pixel Intensity")
hist, _, _ = plt.hist(im_equalized.flatten(), 256, normed=1, facecolor='green')
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.plot(cdf_normalized, color = 'b')

fig = plt.figure()
fig.add_subplot(1,2,1)
plt.title("Contrast Limited Adaptive HE")
imgplot = plt.imshow(im_clahe, cmap='gray', interpolation='nearest')
plt.axis('off')
# Plot histogram
fig.add_subplot(1,2,2)
plt.title("Histogram of Pixels")
plt.xlabel("Pixel Intensity")
hist, _, _ = plt.hist(im_clahe.flatten(), 256, normed=1, facecolor='green')
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.plot(cdf_normalized, color = 'b')


plt.show()