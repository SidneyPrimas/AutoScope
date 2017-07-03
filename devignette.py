"""
    File name: devignette.py
    Author: Sidney Primas
    Date created: 05/07/2017
    Python Version: 2.7
    Description: Remove the illumination pattern in the reversed lens approach. 
"""

# Base packages 
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Import OpenCV
import cv2

# Configuration
source_directory = "./data/20170425/reference/"
final_mask_path = "./data/20170425/reference/illumination_mask.jpg"
image_height = 1944
image_width = 2592

# Obtain images from source_directory
file_list = glob(source_directory + "*.bmp")


im_illumination = np.zeros((image_height, image_width))
for ref_input in file_list: 
	im_single = cv2.imread(ref_input, cv2.IMREAD_GRAYSCALE)
	im_blur = cv2.GaussianBlur(im_single, (15,15), 0)
	im_illumination = np.add(im_blur, im_illumination)



# Scalar division to make illumination mask. Preserves floating point values. 
im_illumination = im_illumination/len(file_list)

# cv2.convertScaleAbs(src[, dst[, alpha[, beta]]]) -> dst
# Converts matrix to an 8-bit data type after scaling by alpha. Turns into int. 
im_illumination = cv2.convertScaleAbs(im_illumination, alpha = 1.0)

# Save illumination mask
cv2.imwrite(final_mask_path, im_illumination, [cv2.IMWRITE_JPEG_QUALITY, 100])



# Implement Illumination Compensation (Demonstration)
# Read Image
im_input = cv2.imread("./data/20170425/reversed_lens/baf3/2.bmp", cv2.IMREAD_GRAYSCALE)
im_final_mask = cv2.imread(final_mask_path, cv2.IMREAD_GRAYSCALE)

# Convert illumination mask into floating point format (to allow for compensation)
im_final_mask = im_final_mask * 1.0

# Compenstate with illumination mask. Still preserves floating point format. 
im_output = np.divide(im_input, im_final_mask)

# cv2.convertScaleAbs(src[, dst[, alpha[, beta]]]) -> dst
# Converts matrix to an 8-bit data type after scaling by alpha. 
im_output = cv2.convertScaleAbs(im_output, alpha = (255.0/np.amax(im_output)))



fig = plt.figure()
plt.title("Illumination Mask")
imgplot = plt.imshow(im_illumination, cmap='gray', interpolation='nearest')
plt.axis('off')

fig = plt.figure()
plt.title("Original")
imgplot = plt.imshow(im_input, cmap='gray', interpolation='nearest')
plt.axis('off')

fig = plt.figure()
plt.title("Illumination Adjusted")
imgplot = plt.imshow(im_output, cmap='gray', interpolation='nearest')
plt.axis('off')

plt.show()