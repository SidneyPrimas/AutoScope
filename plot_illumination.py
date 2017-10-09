"""
    File name: plot_illumination.py
    Author: Sidney Primas
    Date created: 07/04/2017
    Python Version: 2.7
    Description: Plots the intensity of the illumination across the center of the input image. 
"""


# Base packages 
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Import OpenCV
import cv2

# Configuration
source_directory = "./data/20170905/"

# Obtain images from source_directory
file_list = glob(source_directory + "*")

# Iterate through each input file
for ref_input in file_list: 
	im_single = cv2.imread(ref_input, cv2.IMREAD_GRAYSCALE)
	size = im_single.shape

	intensity = im_single[size[0]/5-5:size[0]/5+5, :]
	intensity_avg = np.mean(intensity, axis=0)
	

	im_single[size[0]/5-5:size[0]/5+5, :] = 255

	fig = plt.figure()
	plt.subplot(2, 1, 1)
	name_start = ref_input.rfind('/')
	plt.title("Image: " + ref_input[name_start+1:])
	plt.imshow(im_single, cmap='gray', interpolation='nearest')
	plt.axis('off')

	plt.subplot(2, 1, 2)
	plt.title(ref_input)
	plt.plot(intensity_avg)
	plt.title('Pixel Intensity vs. Pixel Location')
	plt.ylabel('8-Bit Pixel Intensity')
	plt.xlabel('Pixel Location')

plt.show()
