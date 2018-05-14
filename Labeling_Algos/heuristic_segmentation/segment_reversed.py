"""
    File name: segment_reversed.py
    Author: Sidney Primas
    Date created: 06/25/2017
    Python Version: 2.7
    Description: Function to segment an image (from reversed lens) with microbeads. 
"""


# Base packages 
import numpy as np
import matplotlib.pyplot as plt
import time
import json


# Import OpenCV
import cv2


def segmentReversed_Baf3(inputImage_path, outputFolder, target_dim, classes):
	
	fileName_start = inputImage_path.rfind("/")
	input_file_name = inputImage_path[fileName_start+1:-4]
	input_dir_path = inputImage_path[:fileName_start+1]

	# Load image
	im_original = cv2.imread(inputImage_path, cv2.IMREAD_GRAYSCALE)
	im_output = im_original.copy()
	im_orignal_max_px = int(np.amax(im_original))
	im_orignal_min_px = int(np.amin(im_original))

	# Open log file for reading
	log = open(input_dir_path + "coordinates/" + input_file_name + "_coordinates.json" , 'r')
	# Load the data structure saved in JSON format. 
	particle_list = json.load(log)
	log.close()

	# Indicate selected particles 
	for index in range(len(particle_list)):


		# Draw a rectangle around each segmented particle. 
		# Find the corners (opposite sides) of the rectangle. They need to be integer tuples. 
		centroid = particle_list[index]
		x1 = int(centroid[0]-target_dim/2)
		y1 = int(centroid[1]-target_dim/2)
		x2 = int(centroid[0]+target_dim/2)
		y2 = int(centroid[1]+target_dim/2)


		# Applying cropping rectangles to any particle that we detected. 
		#Python: cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> None
		cv2.rectangle(im_output, (x1,y1), (x2,y2), color=im_orignal_max_px, thickness=2)

		# Crop original image to save as segmented particle: im[y1:y2, x1:x2]
		im_cropped_output = im_original[y1:y2, x1:x2]

		### Create dupliate image of original to show segmented particle + context around it (buffer)
		im_original_cpy = im_original.copy()
		# Applying cropping rectangle to image shown to user
		#Python: cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> None
		cv2.rectangle(im_original_cpy, (x1,y1), (x2,y2), color=im_orignal_min_px, thickness=2)

		# Only save partciles that can be cropped into a target_dim square. If it cannot, skipt it. 
		if im_cropped_output.shape != (target_dim, target_dim): 
			continue

		output_image_name = input_file_name + "_" + str(index) + ".jpg"

		# Manual cetegorization
		fig = plt.figure()
		plt.title(output_image_name)
		imgplot = plt.imshow(im_original_cpy, cmap='gray', interpolation='nearest')
		plt.axis('off')
		plt.title(classes)
		# Zoom into correct area of image
		plt.axis([x1-target_dim*4,x2+target_dim*4,y1-target_dim*4,y2+target_dim*4])


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
		cv2.imwrite(dest_path, im_cropped_output, [cv2.IMWRITE_JPEG_QUALITY, 100])


		# Applying cropping rectangles to particles that we selected
		cv2.rectangle(im_output, (x1,y1), (x2,y2), color=im_orignal_min_px, thickness=4)


	# Save input image with selected particles shown
	output_image_name = "%s_selected.jpg"%(input_file_name)
	cv2.imwrite(outputFolder+output_image_name, im_output, [cv2.IMWRITE_JPEG_QUALITY, 100])


	


def segmentReversed_Micro(inputImage, outputFolder, target_dim, classes):

	# Configuration
	min_particle_size = 3 
	mask_path = "./data/20171027/reference/illumination_mask.jpg"

	# Important parameters
	start = inputImage.rfind("/")
	input_file_name = inputImage[start+1:-4]


	### Load Image: Ensure image is in grayscale. 
	im_original = cv2.imread(inputImage, cv2.IMREAD_GRAYSCALE)
	im_output = im_original.copy()
	im_orignal_max_px = int(np.amax(im_original))
	im_orignal_min_px = int(np.amin(im_original))

	### Implement Illumination Compensation
	im_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
	# Convert illumination mask into floating point format (to allow for compensation)
	im_mask = im_mask * 1.0
	# Compenstate with illumination mask. Still preserves floating point format. 
	im_compensated = np.divide(im_original, im_mask)
	# cv2.convertScaleAbs(src[, dst[, alpha[, beta]]]) -> dst
	# Converts matrix to an 8-bit data type after scaling by alpha. 
	im_compensated = cv2.convertScaleAbs(im_compensated, alpha = (255.0/np.amax(im_compensated)))


	### Adaptive threshold: Since we have different illuminations across image, we use an adaptive threshold. 
	# White needs to be the foreground. 
	# The benefit is that adaptiveThreshold is applied to every pixel. So, if a pixel is an outlier, then it will be marked as foreground. 
	# Python: cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) -> dst 
	# Inv Case: Below threshold => white (255). Above threshold => black (0)
	# Regular Case: Below threshold => black (0). Above threshold => white (255)
	# We chose: Small blocksize to capture narrow transition at edge of particle (without having a huge boundary transition). And, not to get confused with multiple particles in block.
	# We chose: Small C so that we have sufficient contrast to definitely get the particle pixels. 
	# Mean (size: 5, C: 5) vs. Guassian (size: 9 since weighted, C: 5): Guassian gives slightly better edge detection since particle around the edge are weighted more heavily. 
	im_thresh = cv2.adaptiveThreshold(im_compensated, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 4)


	### Noise Removal ### => After Thresholding: Use morpholocial transformation (different functions) for the noise removal. 


	### Closing: When we close an image, we remove background pixel from the foreground. 
	# Any background pixel that cannot be contained in any element without touching the foreground is removed. 
	# Closing tries to contain all background pixels in the structured element. 
	# Python: cv2.morphologyEx(src, op, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst
	# Reason: Go from sparse, closely connected pixels to more complete particles. 
	# Reason: Do this with small structures so that only background pixels in the sparse clusters are turned to foreground (and not random noise pixels)
	struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
	im_morph = cv2.morphologyEx(im_thresh, cv2.MORPH_CLOSE, struct_element, iterations = 1)
	# Copy im_morph (so that we can preserve both for graphing)
	im_components = im_morph.copy()


	### Filter markers based on each marker property. 
	# Note: Connectivity indicates how continous 'connected' is defined. If a pixel is not connected on 8 sides, then we do not count it as part of the component. 
	# connectedComponentsWithStats => Gets all the connected components with states on each component
	connected_output = cv2.connectedComponentsWithStats(im_morph, connectivity=8)
	base_num_labels = connected_output[0]
	print "Number of Original Components: %d" % (base_num_labels - 1) # Minus 1 since we don't count the background
	## Extract information from connectedComponents with stats
	base_markers = connected_output[1]
	base_stats = connected_output[2]



	### Description: Loop over each marker. Based on stats, decide to eliminate or include marker. 
	for index in range(base_num_labels):
		# The first label is the background (zero label). We always ignore it. . 
		if index == 0:
			continue

		# Area: If any connected component has an area less than "min_particle_size" pixels, turn it into a background (black)
		if base_stats[index][4] <= min_particle_size:
			im_components[base_markers == index] = 0


	### Notes: After the obvious components have been removed, the rest of the components are consolidated by closing the particles. 
	struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
	im_components_consolidate = cv2.morphologyEx(im_components, cv2.MORPH_CLOSE, struct_element, iterations = 1)

	### Particles + clumps that are close to each other should only be included in a single segmented image. 
	# Essentially, we are including individual and clumps of particles in a single crop. Dilate so clumps/particles merge. 
	struct_element = np.ones((3,3),np.uint8)
	im_components_consolidate = cv2.dilate(im_components_consolidate, struct_element, iterations=5)


	### Find particles (again) using connected components (pixels) approach
	# Note: Connectivity indicates how continous 'connected' is defined. If a pixel is not connected on 8 sides, then we do not count it as part of the component. 
	connected_output = cv2.connectedComponentsWithStats(im_components_consolidate, connectivity=8)
	# Number of lables
	num_labels = connected_output[0]
	print "Number of Final Compondnets: %d" % (num_labels - 1) # Minus 1 since we don't count the background
	# Label matrix (markers)
	im_markers = connected_output[1]
	# Stat matrix: cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA
	stats = connected_output[2]
	# Centroid matrix
	centroids = connected_output[3]


	# Manual Image Segmentation (after identifying the centroids)
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

		# Crop original image to save as segmented particle: im[y1:y2, x1:x2]
		im_cropped_output = im_original[y1:y2, x1:x2]

		### Create dupliate image of original to show segmented particle + context around it (buffer)
		im_original_cpy = im_original.copy()
		# Applying cropping rectangle to image shown to user
		#Python: cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> None
		cv2.rectangle(im_original_cpy, (x1,y1), (x2,y2), color=im_orignal_min_px, thickness=2)



		# Only save partciles that can be cropped into a target_dim square. If it cannot, skipt it. 
		if im_cropped_output.shape != (target_dim, target_dim): 
			continue

		output_image_name = input_file_name + "_" + str(index) + ".jpg"

		# Manual cetegorization
		fig = plt.figure()
		plt.title(output_image_name)
		imgplot = plt.imshow(im_original_cpy, cmap='gray', interpolation='nearest')
		plt.axis('off')
		plt.title(classes)
		# Zoom into correct area of image
		plt.axis([x1-target_dim*4,x2+target_dim*4,y1-target_dim*4,y2+target_dim*4])


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
		cv2.imwrite(dest_path, im_cropped_output, [cv2.IMWRITE_JPEG_QUALITY, 100])


		# Applying cropping rectangles to particles that we selected
		cv2.rectangle(im_output, (x1,y1), (x2,y2), color=im_orignal_min_px, thickness=4)


	# Save input image with selected particles shown
	output_image_name = "%s_selected.jpg"%(input_file_name)
	cv2.imwrite(outputFolder+output_image_name, im_output, [cv2.IMWRITE_JPEG_QUALITY, 100])
