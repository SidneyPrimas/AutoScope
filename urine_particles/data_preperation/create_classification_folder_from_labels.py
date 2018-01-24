# Import basic libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from glob import glob
import json
import sys
import random

# import from local libraries
import utility_functions_data as util
sys.path.insert(0, './urine_particles')
import CNN_functions


"""
Description: Crops the particles from images inraw_image_data based on the ground truth labels in raw_image_data. 

Execution Notes: 
+ Assume standard folder structure: raw_image_data folder is structured so raw_image_data => particle_folders => (original folder + coordinates folder)
+ Crops are taken from images in raw_image_data, and not from the segmentation training/validation folders. 

To Do: 
+ Need to consolidate create_classification_folder_from_labels.py and create_classification_folder_from_model.py. 
++ These two files are based on each other and they have A LOT of overlapping code. 
+ Figure out indication radius + detection radius => Should be the same
"""

# User inputs (apply to any directories)
input_dir_root = './urine_particles/data/clinical_experiment/raw_image_data/'
output_dir_root = './urine_particles/data/clinical_experiment/image_data/20180120_training/'  
classification_folder_name = "classification/"

detection_radius = 30 # Radius (measured in pixels) that indicates the allowable distance between a predicted particle and a reference particle to be deemed accurate. Used on the orignal image
output_crop_size = 64 # The output size of the crops, measured in pixels. Used on the original image. 
validation_proportion = 0.2 #Proportion of images placed in validation
skip_boundary_particles = True # Skip the particles that are on the boundary of the image. 
debug_flag = True


classification_labels = {
	"background"	:	{"background": "other"},
	"10um"			:	{"particle": "10um", "other": "other", "discard": "other", "accident": "other"}, 
	"rbc"			:	{"particle": "rbc", "other": "other", "discard": "other", "accident": "other"}, 
	"wbc"			:	{"particle": "wbc", "other": "other", "discard": "other", "accident": "other"}, 
}
nclasses = 4


""" Fixed + auto-calculated parameters. """
classification_folder_dir = output_dir_root + classification_folder_name
training_root_dir = classification_folder_dir + "training/"
validation_root_dir = classification_folder_dir + "validation/"
classification_debug_output = classification_folder_dir + "debug_output/"
segmentation_metadata_log = output_dir_root + "segmentation_metadata.log"
classification_metadata_path = output_dir_root + 'classification_metadata.log'
segmentation_metadata = None # updated during execution 




def main():


	# Build folders
	create_classification_subfolder(training_root_dir, classification_labels)
	if (debug_flag):
		create_classification_subfolder(classification_debug_output, classification_labels)


	# Get segmentation metadata
	log = open(segmentation_metadata_log, 'r')
	global segmentation_metadata # Defined as global since update variable in global context during execution.
	segmentation_metadata = json.load(log)
	log.close()


	# Generate all the crops for classification (where each class is placed in a different folder)
	generate_crops_from_labels(segmentation_metadata)

	# Split into validation and training data
	util.split_data(input_dir=training_root_dir, output_dir=validation_root_dir, move_proportion=validation_proportion)

	# Create classifcation metadata log
	create_classification_metadata_log(classification_metadata_path)


def generate_crops_from_labels(segmentation_metadata):
	"""Description: Crop particles based on ground truth labels."""

	# Creates crop for reference coordinate log in in raw_image_data.
	for particle_folder in segmentation_metadata["input_particle_folders"]:

		coordinate_files = glob(input_dir_root + particle_folder + "coordinates/*_classes.json")

		for coordinate_log_path in coordinate_files:

			# Obtain original image from from the raw_image_data folder
			base_file_name, particle_folder = util.get_image_name_from_coordinate_log(coordinate_log_path)
			original_img_path = input_dir_root + particle_folder + "/original/" + base_file_name + ".bmp"
			original_img = cv2.imread(original_img_path)

			# Obtain data from coordinate log. 
			log = open(coordinate_log_path, 'r')
			coordinate_log_data = json.load(log)
			particle_list = coordinate_log_data["particle_list"]
			log.close()

			# Crop the original image based on the predicted segmentation. Label the crops based on the reference coordinates. 
			original_img_cropped = crop_particles_into_class_folders_using_labels(original_img, particle_list)

			# Save output with cropped images
			if (debug_flag):
				debug_img_output_path = classification_debug_output + particle_folder + "/predicted_particles_" + base_file_name + ".bmp"
				cv2.imwrite(debug_img_output_path, original_img_cropped)


def crop_particles_into_class_folders_using_labels(original_image, particle_list):
	"""
	Description: Crop particles based on ground truth labels. 
	"""
	original_cpy = None
	if (debug_flag):
		original_cpy = original_image.copy()

	# Turn reference coordinates into numpy array (useful for matrix calculation)
	coordinates_truth_list, metadata_truth_list = map(list, zip(*particle_list))
	coordinates_truth_list = np.asarray(coordinates_truth_list)

	# Crop each labeled particle from the original image. 
	for index, centroid in enumerate(coordinates_truth_list): 

		target_truth_metadata = metadata_truth_list[index]
		class_name = util.metadata_to_label(target_truth_metadata, classification_labels)

		# Crop Original image
		x1, x2, y1, y2 = CNN_functions.get_crop_coordinates(original_image.shape, centroid, output_crop_size)
		crop = original_image[y1:y2,x1:x2]

		# Skip particles that are at the edge of the frame. 
		if (skip_boundary_particles):
			if (crop.shape[0] != output_crop_size) or (crop.shape[1] != output_crop_size): 
				continue

		# Save cropped image
		class_folder = class_name + "/"
		file_name = target_truth_metadata["main"] + "_" + target_truth_metadata["sub"] + "_" + str(int(centroid[0])) + "_" + str(int(centroid[1])) + ".bmp"
		output_path = training_root_dir + class_folder + file_name
		cv2.imwrite(output_path, crop)

		# Place indicator for each ground truth partidcle detected. 
		if (debug_flag): 
			circl_centroid = (int(centroid[0]), int(centroid[1]))
			cv2.circle(
				original_cpy, 
				center=circl_centroid, 
				radius=segmentation_metadata["indicator_radius"], 
				color=(0, 255, 0), 
				thickness=2)

	# Return original_cpy for debugging purposes
	return original_cpy



def create_classification_subfolder(path, classification_labels_dict):
	""" Builds the classification folder, with the appropriate structure."""

	# If the 'classification' folder exists, raise an error. 
	if (os.path.isdir(path)): 
		raise UserWarning("Class folder already created. Output folders for each class should not exist yet.")
	os.makedirs(path)

	create_subfolder_struct(path, classification_labels_dict)



def create_subfolder_struct(target_dir, classification_labels_dict):
	"""
	Makes class folders in path directory. 
	"""
	# Build the class folders based on classification_labels_dict
	for _, subclass_data in classification_labels_dict.iteritems():
		for _, target_class in subclass_data.iteritems():
			path = target_dir + target_class
			if (not os.path.isdir(path)): # Only create if the class doesn't exist yet. 
				os.mkdir(path)


def create_classification_metadata_log(path):

	classification_metadata = {
		"classification_labels": classification_labels, 
		"nclasses": nclasses, 
		"detection_radius": detection_radius, 
		"output_crop_size": output_crop_size, 
		"skip_boundary_particles": skip_boundary_particles, 
		"validation_proportion": validation_proportion, 
		"segmentation_metadata": segmentation_metadata
	}

	
	output_log = open(path , 'w+')
	json.dump(classification_metadata, output_log)
	output_log.close()




if __name__ == "__main__":
	main()