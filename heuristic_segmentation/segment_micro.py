"""
    File name: segment_micro.py
    Author: Sidney Primas
    Date created: 05/03/2017
    Python Version: 2.7
    Description: Function to segment an image (from microscope)
"""


# Base packages 
import numpy as np
import matplotlib.pyplot as plt
import time


# Import OpenCV
import cv2



def segmentMicroImage(inputImage, outputFolder, target_dim, classes):

	# Configuration
	open_kern_size = 3 # 10um => 7; 6um => 5; baf3 => 3

	# Important parameters
	start = inputImage.rfind("/")
	input_file_name = inputImage[start+1:-4]


	# Load Image: Ensure image is in grayscale. 
	im_original = cv2.imread(inputImage, cv2.IMREAD_GRAYSCALE)
	im_output = im_original.copy()
	im_orignal_max_px = int(np.amax(im_original))
	im_orignal_min_px = int(np.amin(im_original))


	# Adaptive threshold: Since we have different illuminations across image, we use an adaptive threshold 
	# The benefit is that adaptiveThreshold is applied to every pixel. So, if a pixel is an outlier, then it will be marked as foreground. 
	#im_blur = cv2.GaussianBlur(im_original, (15,15), 0)
	# Python: cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) -> dst 
	im_thresh = cv2.adaptiveThreshold(im_original, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 5)


	# Noise Removal: Use morpholocial transformation (different functions) for th noise removal. 
	# Closing: When we close an image, we remove background pixel from the foreground. 
	# Any background pixel that cannot be contained in any element without touching the foreground is removed. 
	# Python: cv2.morphologyEx(src, op, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst
	#struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2) )
	#im_morph = cv2.morphologyEx(im_thresh, cv2.MORPH_CLOSE, struct_element, iterations = 1)

	# Opening: When we open, we remove foreground pixel. Any foreground pixel that cannot be contained in the the element without touching the background.
	struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kern_size,open_kern_size) )
	im_morph = cv2.morphologyEx(im_thresh, cv2.MORPH_OPEN, struct_element, iterations = 2) 

	# Finding particles (individual and clumps of particles) to crop
	struct_element = np.ones((3,3),np.uint8)
	sure_bg = cv2.dilate(im_morph, struct_element, iterations=5)


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


	# Crop images
	# Create figure to categorize particles
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

		### Create dupliate image of original to show segmented particle + context around it (buffer)
		im_original_cpy = im_original.copy()
		# Applying cropping rectangle to image shown to user
		#Python: cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> None
		cv2.rectangle(im_original_cpy, (x1,y1), (x2,y2), color=im_orignal_min_px, thickness=2)

		# Only save partciles that can be cropped into a target_dim square. If it cannot, skipt it. 
		if im_cropped.shape != (target_dim, target_dim): 
			continue

		output_image_name = input_file_name + "_" + str(index) + ".jpg"

		# Manual cetegorization
		fig = plt.figure()
		plt.title(output_image_name)
		imgplot = plt.imshow(im_original_cpy, cmap='gray', interpolation='nearest')
		plt.axis('off')
		plt.title(classes)
		# Zoom into correct area of image
		plt.axis([x1-target_dim,x2+target_dim,y1-target_dim,y2+target_dim])


		plt.show(block=False)
		keypress = raw_input(">")

		# Categorize the image based on user keypress
		if keypress == "s": # skip particle
			plt.close(fig)
			continue
		elif keypress == "break": # break out of image. Go to next image. 
			plt.close(fig)
			break
		elif keypress in classes:
			dest_path = outputFolder + classes[keypress] + "/" + output_image_name
		# If the keypress isn't valid, then place the picture in other. 
		else:
  			dest_path = outputFolder + classes["0"] + "/" + output_image_name

		plt.close(fig)
		
		# Save cropped image
		print dest_path
		cv2.imwrite(dest_path, im_cropped, [cv2.IMWRITE_JPEG_QUALITY, 100])


		# Applying cropping rectangles to particles that we selected
		cv2.rectangle(im_output, (x1,y1), (x2,y2), color=im_orignal_min_px, thickness=4)


	# Save input image with selected particles shown
	output_image_name = "%s_selected.jpg"%(input_file_name)
	cv2.imwrite(outputFolder+output_image_name, im_output, [cv2.IMWRITE_JPEG_QUALITY, 100])

