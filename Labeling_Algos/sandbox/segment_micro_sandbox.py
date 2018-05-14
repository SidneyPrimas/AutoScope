"""
    File name: segmentation_micro_sandbox.py
    Author: Sidney Primas
    Date created: 05/02/2017
    Python Version: 2.7
    Description: Laboratory for figuring out segmentation. 
"""


# Base packages 
import numpy as np
import matplotlib.pyplot as plt


# Import OpenCV
import cv2
# Argparse: For command line python arguments
import argparse

# Construct the argument parse and parse the arguments
# -i: Path to image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# Load Image: Ensure image is in grayscale. 
im_original = cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE)
im_output = im_original.copy()
im_orignal_max_px = int(np.amax(im_original))
im_orignal_min_px = int(np.amin(im_original))
target_dim = 52

# Rescale pixel intensities to have a max value of 255. 
#im_original = cv2.convertScaleAbs(im_original, alpha = (255.0/np.amax(im_original)))

# Includes Otsu method that selects threshold automatically based on bimodal distribution. 
# This only works if there is a strong bimodal distribution. 
# Threshold:cv2.threshold(src, thresh, maxval (set to value), type) -> threshold_value, image_dst
#ret_thresh, im_thresh = cv2.threshold(im_original, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Apply Guassian blur in order to reduce noise in background. 
# Not needed because open/close morphological functions allow us to do this as well. 
# Python: cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) -> dst
# By setting the sigma to 0, the sigma is auto calculate. 
# im_blur = cv2.GaussianBlur(im_original, (15,15), 0)


# Adaptive threshold: Since we have different illuminations across image, we use an adaptive threshold 
# The benefit is that adaptiveThreshold is applied to every pixel. So, if a pixel is an outlier, then it will be marked as foreground. 
# Python: cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) -> dst 
im_thresh = cv2.adaptiveThreshold(im_original, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 5)


# Noise Removal: Use morpholocial transformation (different functions) for th noise removal. 

# Closing: When we close an image, we remove background pixel from the foreground. 
# Any background pixel that cannot be contained in any element without touching the foreground is removed. 
# Python: cv2.morphologyEx(src, op, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst
#struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2) )
#im_morph = cv2.morphologyEx(im_thresh, cv2.MORPH_CLOSE, struct_element, iterations = 1)

# Opening: When we open, we remove foreground pixel. Any foreground pixel that cannot be contained in the the element without touching the background.
# For 10um: struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7) )
struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7) )
im_morph = cv2.morphologyEx(im_thresh, cv2.MORPH_OPEN, struct_element, iterations = 2) 

# Getting the cell outlines
struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2) )
im_outlines = cv2.morphologyEx(im_morph, cv2.MORPH_GRADIENT, struct_element, iterations = 1) 

# Finding 'sure background' area
struct_element = np.ones((3,3),np.uint8)
sure_bg = cv2.dilate(im_morph, struct_element, iterations=5)

# Finding 'sure foreground' area
# Note: We have the option to use erosion. However, Erosion doesn't work well with overlapping or connected cells. 
# sure_fg = cv2.erode(im_morph, struct_element, iterations=3)
# Instead, use the distance transform. This will help you find the peak of each cell, and 70% around the peak. The rest is removed. 
# Python: cv2.distanceTransform(src, distanceType, maskSize[, dst]) -> dst
# Note: In order to get the Euclidian distance, we use the cv2.DIST_L2 (which assigns costs to each distance shift). 
# Note: We use a 3x3 kernel shift to calculate distance. 
dist_transform = cv2.distanceTransform(im_morph, cv2.DIST_L2, 3)
# Note, we use 70% of the max distnace as the reference for the threshold.
# Note: Use Thresh_binary directly, since we already used inverted thresh previously 
_, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, cv2.THRESH_BINARY)


# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)


# Create marker mask
# Note: Connectivity indicates how continous 'connected' is defined. If a pixel is not connected on 8 sides, then we do not count it as part of the component. 
connected_output = cv2.connectedComponentsWithStats(sure_bg, connectivity=8)
# Number of lables
num_labels = connected_output[0]
print "Number of Compondnets: %d" % (num_labels - 1) # Minus 1 since we don't count the background
# Label matrix (markers)
im_markers = connected_output[1]
# Stat matrix: cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA
stats = connected_output[2]
print "Order of Stats Matrix: cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA"
print stats
# Centroid matrix
centroids = connected_output[3]


# Segment images
for index in range(num_labels):
	# The first label is the background (zero label). We ignore it. 
	if index == 0:
		continue

	# Draw a rectangle around each segmented particle. 
	# Find the corners (opposite sides) of the rectangle. They need to be integer tuples. 
	centroid = centroids[index]
	x1 = int(centroid[0]-target_dim/2)
	y1 = int(centroid[1]-target_dim/2)
	x2 = int(centroid[0]+target_dim/2)
	y2 = int(centroid[1]+target_dim/2)


	# Applying cropping rectangles to any particle that we detected. 
	#Python: cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> None
	cv2.rectangle(im_output, (x1,y1), (x2,y2), color=im_orignal_max_px, thickness=2)

	# Crop the Original Image: im[y1:y2, x1:x2]
	im_cropped = im_original[y1:y2, x1:x2]

	# Only save partciles that can be cropped into a target_dim square. If it cannot, skipt it. 
	if im_cropped.shape != (target_dim, target_dim): 
		continue

	output_path = "./data/20170425/particles_output/%d.jpg" % (index)
	cv2.imwrite(output_path, im_cropped, [cv2.IMWRITE_JPEG_QUALITY, 100])

	# Applying cropping rectangles to particles that we selected
	cv2.rectangle(im_output, (x1,y1), (x2,y2), color=im_orignal_min_px, thickness=4)





#### IMAGES ####
fig = plt.figure()
plt.title("Original")
imgplot = plt.imshow(im_original, cmap='gray', interpolation='nearest')
plt.axis('off')


fig = plt.figure()
plt.title("Threshold")
imgplot = plt.imshow(im_thresh, cmap='gray', interpolation='nearest')
plt.axis('off')

fig = plt.figure()
plt.title("Morpholocial")
imgplot = plt.imshow(im_morph, cmap='gray', interpolation='nearest')
plt.axis('off')

fig = plt.figure()
fig.add_subplot(3,1,1)
plt.title("Black Region Shows 'Sure Background'")
imgplot = plt.imshow(sure_bg, cmap='gray', interpolation='nearest')
plt.axis('off')
fig.add_subplot(3,1,2)
plt.title("White Region Shows 'Sure Foreground'")
imgplot = plt.imshow(sure_fg, cmap='gray', interpolation='nearest')
plt.axis('off')
fig.add_subplot(3,1,3)
plt.title("White Region Shows 'Unknown Region'")
imgplot = plt.imshow(unknown, cmap='gray', interpolation='nearest')
plt.axis('off')

fig = plt.figure()
plt.title("Distance Transform")
imgplot = plt.imshow(dist_transform, interpolation='nearest')
plt.axis('off')

fig = plt.figure()
plt.title("Markers")
imgplot = plt.imshow(im_markers, interpolation='nearest')
plt.axis('off')

fig = plt.figure()
plt.title("Final Output")
imgplot = plt.imshow(im_output, cmap='gray', interpolation='nearest')
plt.axis('off')

plt.show()