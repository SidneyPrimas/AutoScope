"""
    File name: plot_illumination.py
    Author: Sidney Primas
    Date created: 10/08/2017
    Python Version: 2.7
    Description: Plots the intensity of the illumination across the center of the input image. 
"""


# Base packages 
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import math

# Import OpenCV
import cv2

# Configuration
source_directory = "./data/meeting/"

# Obtain images from source_directory
file_list = glob(source_directory + "*")

# Iterate through each input file
for ref_input in file_list: 
	im_original = cv2.imread(ref_input, cv2.IMREAD_GRAYSCALE)
	size = im_original.shape

	#### LINEAR VISUALIZATIONS ###
	im_linear = im_original.copy()
	intensity = im_linear[size[0]/5-5:size[0]/5+5, :]
	intensity_avg = np.mean(intensity, axis=0)
	im_linear[size[0]/5-5:size[0]/5+5, :] = 255

	### MEASURING MIN/MAX INTENSITY ###
	im_range = im_original.copy()
	# FILTERING
	# gaussian vs blur vs median filter: The gaussian filter attenuates outlier pixels the least while the median filter attenuates them the most. 
	# The goal is to remove outliers pixels from a kernel. Thus, the median filter is the most appropriate. 
	# However, if the median filter and the gaussian/blur filter give significantly different results, that needs to be investigated. 

	# Blur Filter
	# Reference: cv2.blur(src, ksize[, dst[, anchor[, borderType]]])
	# To handle edge cases, reflect image around borders (ignoring the first column/row)
	# Use large kernel sizes since really care about broad illumination pattern (not pixel to pixel variation)
	im_range_blur = cv2.blur(src=im_range, ksize=(20,20), borderType= cv2.BORDER_REFLECT_101)

	# Median Filter
	# Reference: cv2.medianBlur(src, ksize[, dst])
	im_range_median= cv2.medianBlur(src=im_range, ksize=21)


	#### 3D Visualization ####
	# Resize the image so that it's more reasonable to work with. 
	# Another option to resize: scipy.misc.imresize(image, 0.5)
	im_range_median_small = cv2.resize(im_range_median, (0,0), fx=0.1, fy=0.1) 

	# create the x and y coordinate arrays (here we just use pixel indices)
	mesh_xx, mesh_yy = np.mgrid[0:im_range_median_small.shape[0], 0:im_range_median_small.shape[1]]

	#### PLOTS ####

	# Linear Plot
	fig = plt.figure()
	plt.subplot(2, 1, 1)
	name_start = ref_input.rfind('/')
	plt.title("Image: " + ref_input[name_start+1:])
	plt.imshow(im_linear, cmap='gray', interpolation='nearest')
	plt.axis('off')

	plt.subplot(2, 1, 2)
	plt.title(ref_input)
	plt.plot(intensity_avg)
	plt.title('Pixel Intensity vs. Pixel Location')
	plt.ylabel('8-Bit Pixel Intensity')
	plt.xlabel('Pixel Location')

	# Blurred Image
	fig = plt.figure()
	plt.subplot(2, 2, 1)
	plt.imshow(im_original, cmap='gray', interpolation='nearest')
	plt.title('Original')
	plt.axis('off')
	plt.subplot(2, 2, 2)
	plt.imshow(im_range_blur, cmap='gray', interpolation='nearest')
	plt.title('Blur')
	plt.axis('off')
	plt.subplot(2, 2, 3)
	plt.imshow(im_range_median, cmap='gray', interpolation='nearest')
	plt.title('Median')
	plt.axis('off')

	# Histogram
	fig = plt.figure()
	fig.add_subplot(1,2,1)
	plt.title("Original Image")
	plt.imshow(im_range_median, cmap='gray', interpolation='nearest')
	plt.axis('off')

	# Plot histogram
	## normed indicates that the integral of the histogram is 1
	## flatten(): Flattens the image into a single array
	fig.add_subplot(1,2,2)
	plt.title("Histogram of Pixels")
	plt.xlabel("Pixel Intensity")

	im_range_median_flat = im_range_median.flatten()
	hist, _, _ = plt.hist(im_range_median_flat, np.arange(256), normed=1, facecolor='green')

	im_range_median_sorted = np.sort(im_range_median_flat)

	quartile_1_value = im_range_median_sorted[int(im_range_median_sorted.size*.01)]
	quartile_4_value = im_range_median_sorted[int(im_range_median_sorted.size*.99)]
	delta =  quartile_4_value-quartile_1_value
	percent_of_range = 100.0*(delta/256.0)

	# Get the cumulative sum of the histogram.
	cdf = hist.cumsum()
	# Normalize so that the CDF is represented on same scale as histogram
	cdf_normalized = cdf * hist.max()/ cdf.max()
	plt.plot(cdf_normalized, color = 'b')
	plt.bar([quartile_1_value, quartile_4_value], [hist.max(),hist.max()], color = 'r')

	summary = "5th Percentile: %d, 95th Percentile: %d, Range: %d, Percent Change: %f"%(quartile_1_value, quartile_4_value, delta, percent_of_range)
	print ref_input
	print summary

	# 3D Visualizations
	# create the figure
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_surface(mesh_xx, mesh_yy, im_range_median_small ,rstride=1, cstride=1, cmap=plt.cm.gray,linewidth=0)


plt.show()
