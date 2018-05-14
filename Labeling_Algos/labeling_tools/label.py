# Base packages 
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import cv2
from glob import glob
import random
import os
from datetime import datetime

# Local libraries
import utility_functions

"""
Execution Notes: 
+ Need to create (manually) the subfulder structure with the following subfolders: annotations_output, coordinates, original, overlay_output. 
+ If the particles have been identified and classified, use talbe_typ = "class". 
++ If you want to just label with foreground/background, use type_to_label to indicate foreground/background. 
++ Only use label_foreground() when particle coordinates have been indicated without the corresponding class. 
"""
root_dir = "./data/20180120/"
label_type = "classes" # select between 'classes' for labeling each pixel a class and 'foreground' for binary labeling. 
indicator_radius = 15
crop_images_flag = False

# Path to the subfolder. Fixed structure of subfolder: 
# + Subfolder
# ++ annotations_output
# ++ coordinates
# ++ original
# ++ overlay_output
subfolders = [
	root_dir + "wbc_full_rev1/selected/", 
	root_dir + "10um_rev1/selected/", 
	root_dir + "rbc_half_rev1/selected/", 
]

# Implementation Notes: Labels need to start at 0 (with the bacgkround), and increment monotonically by 1. 
# Structure 1) key => main category : {key => sub-category}
# Structure 2) key => folder_name 	: {key => sub-category}
type_to_label = {
	"background"	:	{"background": 0} ,# background always has a 0 label
	"10um"			:	{"particle": 1, "other": 4, "discard": 4, "accident": 4}, 
	"rbc"			:	{"particle": 2, "other": 4, "discard": 4, "accident": 4}, 
	"wbc"			:	{"particle": 3, "other": 4, "discard": 4, "accident": 4}, 
}
nclasses = 5 # Total number of classes, including the background class. 



coordinates_extension = "*_classes.json" if label_type == "classes" else "*_coordinates.json"  # "*_coordinates.json" or "*_classes.json"

# Crop Parameters 
crop_increment = 120 # How much the crop window is shifted for each crop 
crop_size = 480 # The size of the crop: crop_size x crop_size 

def main(): 

	for target_subfolder in subfolders:
		coordinate_folder = target_subfolder + "coordinates/"
		coordinate_files = glob(coordinate_folder + coordinates_extension)
		for log_path in coordinate_files:

			if (label_type == "classes"):
				label_classes(log_path)

			if (label_type == "foreground"):
				label_foregound(log_path)

			if (crop_images_flag): 
				output_folder = label_type + "/" # The name of the output folder within original and annotations_output 
				utility_functions.crop_images(log_path, output_folder , crop_increment, crop_size)


	# Dump data to json
	json_data = {
		"class_labels": type_to_label, 
		"nclasses": nclasses, 
		"subfolders": subfolders
	}
	file_name = datetime.strftime(datetime.now(), 'data_config_%Y%m%d_%H-%M-%S.log')
	output_log_path = root_dir + file_name
	output_log = open(output_log_path , 'w+')
	json.dump(json_data, output_log)
	output_log.close()

def label_foregound(log_path):
	"""
	Use label_foreground when the coordinates of each particle have been recorded, but the particles have not been manually classified. 
	"""
	# Define directory/file names
	image_file_name, subfolder = utility_functions.get_image_file_name(log_path)

	# Make necessary folders
	output_image_annotated_class_dir = subfolder + "annotations_output/foreground/"
	if (not os.path.isdir(output_image_annotated_class_dir)):
		os.makedirs(output_image_annotated_class_dir)

	# Define output paths
	input_image_path = subfolder + "original/" + image_file_name + ".bmp"
	output_image_annotated_path = output_image_annotated_class_dir + image_file_name + ".bmp"
	output_image_overlay_path = subfolder + "overlay_output/"  + image_file_name + "_overlay_coordinates.bmp"
	output_image_display_path = subfolder + "overlay_output/"  + image_file_name + "_display_coordinates.bmp"

	# Load image
	im_overlay = cv2.imread(input_image_path) # Careful: format in BGR
	im_mask = np.zeros(im_overlay.shape)
	im_mask_display = np.zeros(im_overlay.shape)

	# Load the coordinate data structure saved in JSON format. 
	log = open(log_path, 'r')
	particle_list = json.load(log)
	log.close()

	# Label foreground pixels
	for index in range(len(particle_list)):

		centroid = particle_list[index]
		centroid = (int(centroid[0]), int(centroid[1]))
		cv2.circle(im_overlay, center=centroid, radius=indicator_radius, color=(255,255,255), thickness=-1) # Label pixels on overlay image
		cv2.circle(im_mask, center=centroid, radius=indicator_radius, color=(1,1,1), thickness=-1) # Label pixels on mask image
		cv2.circle(im_mask_display, center=centroid, radius=indicator_radius, color=(255,255,255), thickness=-1) # Label pixels on mask image

	# Save input image with selected particles shown
	cv2.imwrite(output_image_overlay_path, im_overlay)
	cv2.imwrite(output_image_annotated_path, im_mask)
	cv2.imwrite(output_image_display_path, im_mask_display)


def label_classes(log_path):
	"""
	Use label_classes when the coordinates of each particle have been identified, and the particle has been manually classified. 
	For all practical purposes, we use label_classes. 
	"""

	# Define directory/file names
	image_file_name, subfolder = utility_functions.get_image_file_name(log_path)

	# Make necessary folders
	output_image_annotated_class_dir = subfolder + "annotations_output/classes/"
	if (not os.path.isdir(output_image_annotated_class_dir)):
		os.makedirs(output_image_annotated_class_dir)

	# Define output paths
	input_image_path = subfolder + "original/" + image_file_name + ".bmp"
	output_image_annotated_path = output_image_annotated_class_dir + image_file_name + ".bmp"
	output_image_overlay_path = subfolder + "overlay_output/"  + image_file_name + "_overlay_classes.bmp"
	output_image_display_path = subfolder + "overlay_output/"  + image_file_name + "_display_classes.bmp"


	# Load image
	im_overlay = cv2.imread(input_image_path) # Careful: format in BGR
	im_mask = np.zeros(im_overlay.shape)
	im_mask_display = np.zeros(im_overlay.shape)

	# Load the coordinate data structure saved in JSON format. 
	log = open(log_path, 'r')
	json_data = json.load(log)
	log.close()

	# Extract json data
	particle_list = json_data['particle_list']
	classes = json_data['classes']

	verify_class_configuration(classes_struct= classes)


	# Preselect random colors
	colors = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(nclasses-1)]

	for index in range(len(particle_list)):
		#colors = [(0,0,255), (0,255,0), (255,0,0), (255,255,255)]
		centroid = particle_list[index][0]
		class_metadata = particle_list[index][1]
		pixel_label = metadata_to_label(class_metadata)
		centroid = (int(centroid[0]), int(centroid[1]))
		indicator_color = colors[pixel_label - 1] # "-1" since first non-background label is 1. 
		class_color = (pixel_label, pixel_label, pixel_label)

		# Add indicators to image. 
		cv2.circle(im_overlay, center=centroid, radius=indicator_radius, color=indicator_color, thickness=3) # Label pixels on overlay image
		cv2.circle(im_mask, center=centroid, radius=indicator_radius, color=class_color, thickness=-1) # Label pixels on mask image (class color)
		cv2.circle(im_mask_display, center=centroid, radius=indicator_radius, color=indicator_color, thickness=-1) # Label pixels on mask (variable color)

	# Save input image with selected particles shown
	cv2.imwrite(output_image_overlay_path, im_overlay)
	cv2.imwrite(output_image_annotated_path, im_mask)
	cv2.imwrite(output_image_display_path, im_mask_display)



def verify_class_configuration(classes_struct):
	for _, class_metadata in classes_struct.iteritems(): 
		main_class = class_metadata["main"]
		sub_class = class_metadata["sub"]
		if main_class in type_to_label:
			if sub_class in type_to_label[main_class]:
				continue

		raise UserWarning("Key not in typte_to_labe struct.") 

def metadata_to_label(class_metadata): 
	main_class = class_metadata["main"]
	sub_class = class_metadata["sub"]

	label = type_to_label[main_class][sub_class]

	return label



if __name__ == "__main__":
    main()
