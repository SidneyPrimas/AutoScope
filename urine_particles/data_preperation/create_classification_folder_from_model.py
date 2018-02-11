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


#Import keras libraries
from tensorflow.python.keras.optimizers import SGD, RMSprop
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.utils import plot_model

# import from local libraries
import utility_functions_data as util
sys.path.insert(0, './urine_particles')
from SegmentParticlesData import SegmentParticlesData
import CNN_functions
from model_FCN8 import FCN8_32px_factor as createModel
from SegmentParticles_config import SegmentParticles_Config

"""
Description: Crops the particles from images in raw_image_data based on the segmentation location predicted by the model. 
Goal of script: The goal of this script is to understand the reduction in classification accuracy when... 
	the canvas image is cropped based on predicted particles locations vs. actual particle locations (in create_classification_folder_from_model.py)...
	Obviously, in the end-to-end system, particles are cropped based on the model. 

Execution Notes: 
+ Assume standard folder structure: raw_image_data folder is structured so raw_image_data => particle_folders => (original folder + coordinates folder)
+ Crops are taken from images in raw_image_data, and not from the segmentation training/validation folders. 

To Do: 
+ Possible To Do: Integrate + refactor code with create_classification_folder_from_model.py
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
	generate_crops_from_model(segmentation_metadata)

	# Split into validation and training data
	util.split_data(input_dir=training_root_dir, output_dir=validation_root_dir, move_proportion=validation_proportion, in_order=False)


	# Augment the dataset so each class has equal images
	util.balance_classes_in_dir(input_dir=training_root_dir)
	util.balance_classes_in_dir(input_dir=validation_root_dir)

	
	# Create classifcation metadata log
	create_classification_metadata_log(classification_metadata_path)




def generate_crops_from_model(segmentation_metadata):
	"""
	Description: Crop particles based on segmentations produced by model. 
	"""
	# Build the semantic segmentation model. 
	model, data = initialize_segmentation_model()

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

			# Create reference labeled image. 
			im_mask = np.zeros(original_img.shape) 
			label_colors = [(i,i,i) for i in range(segmentation_metadata["nclasses"])]
			im_mask = util.label_single_image(
				input_image=im_mask, 
				particle_list=particle_list, 
				color_list=label_colors, 
				radius=segmentation_metadata["indicator_radius"], 
				segmentation_labels=segmentation_metadata["segmentation_labels"])

		
			# Predict segmentation of original_img with segmentation model  
			temp_annotations_path = classification_folder_dir + "temp.bmp"
			cv2.imwrite(temp_annotations_path, im_mask)
			label_pred, label_truth = data.validate_image(model, original_img_path, temp_annotations_path)
			os.remove(temp_annotations_path) # Remove the temp file. 

			# Determine particle_accuracy
			if (debug_flag):
				save_path = classification_debug_output + particle_folder + "/segmentation_accuracy" + base_file_name
				CNN_functions.get_foreground_accuracy_perImage(
					truth_array = label_truth, 
					pred_array = label_pred, 
					config = data.config, 
					radius = detection_radius,
					base_output_path = save_path)

			# Reshape the predicted labels.
			label_pred_reshaped = np.argmax(label_pred, axis=1) # Convert from categorical format to label format. 
			label_pred_reshaped = np.reshape(label_pred_reshaped, data.config.target_size)

			# Crop the original image based on the predicted segmentation. Label the crops based on the reference coordinates. 
			original_img_cropped = crop_particles_into_class_folders_using_model(original_img, label_pred_reshaped, particle_list)

			# Save output with cropped images
			if (debug_flag):
				debug_img_output_path = classification_debug_output + particle_folder + "/predicted_particles_" + base_file_name + ".bmp"
				cv2.imwrite(debug_img_output_path, original_img_cropped)



def crop_particles_into_class_folders_using_model(original_image, predicted_image, particle_list):
	"""
	Description: Crop particles based on segmentations produced by model. 
	"""
	# Obtain scaling factors between orginal and predicted images. 
	factor_height, factor_width = get_scale_factors(original_image, predicted_image)
	
	original_cpy = None
	if (debug_flag):
		original_cpy = original_image.copy()


	# Get the centroids of each segmented particle
	pred_centroids_list = get_segmentation_coordinates(predicted_image)

	# Turn reference coordinates into numpy array (useful for matrix calculation)
	coordinates_truth_list, metadata_truth_list = map(list, zip(*particle_list))
	coordinates_truth_list = np.asarray(coordinates_truth_list)

	# For each predicted particle, determine the label of the particle based on the reference coordinates. 
	for index_pred, centroid_pred in enumerate(pred_centroids_list): 

		# Rescale centroid_pred to match with original data. 
		# centroid_pred is [width, height] format.
		centroid_pred_upscaled = scale_centroid(centroid_pred, [1/factor_width, 1/factor_height])


		# Compare the upscaled predicted centroid with the ground truth centroids. 
		# Nearest distance is on the scale of original_image. 
		nearest_i_truth, nearest_distance = CNN_functions.nearest_centroid(centroid_pred_upscaled, coordinates_truth_list)


		# Label each segmented particle in the predicted set. 
		target_truth_metadata = metadata_truth_list[nearest_i_truth]
		if nearest_distance < detection_radius:
			class_name = util.metadata_to_label(target_truth_metadata, classification_labels)
		else: # If the predicted particle has no associated reference class, put it in the 'other' category. 
			class_name = "other" 


		# Crop Original image
		x1, x2, y1, y2 = CNN_functions.get_crop_coordinates(original_image.shape, centroid_pred_upscaled, output_crop_size)
		crop = original_image[y1:y2,x1:x2]

		# Skip particles that are at the edge of the frame. 
		if (skip_boundary_particles):
			if (crop.shape[0] != output_crop_size) or (crop.shape[1] != output_crop_size): 
				continue


		# Save cropped image
		class_folder = class_name + "/"
		# Important: Order of centroid data important! Used as a feature later. 
		file_name = target_truth_metadata["main"] + "_" + target_truth_metadata["sub"] + "_" + str(int(centroid_pred_upscaled[0])) + "_" + str(int(centroid_pred_upscaled[1])) + ".bmp"
		output_path = training_root_dir + class_folder + file_name
		cv2.imwrite(output_path, crop)

		# Place indicator for each ground truth partidcle detected. 
		if (debug_flag): 
			circl_centroid = (int(centroid_pred_upscaled[0]), int(centroid_pred_upscaled[1]))
			cv2.circle(
				original_cpy, 
				center=circl_centroid, 
				radius=segmentation_metadata["indicator_radius"], 
				color=(0, 255, 0), 
				thickness=2)

	# Return original_cpy for debugging purposes
	return original_cpy
	

def scale_centroid(centroid, factor):
	"""
	Scale the x/y of the centroid by factor. 
	centroid: list/tuple of centroid [x, y]
	factor:list/tuple of factor [x_factor, y_factor]
	scaled_centroid: the scaled centroid [x*x_factor, y*y_factor]
	"""
	scaled_centroid = [centroid[0]*factor[0], centroid[1]*factor[1]]
	return scaled_centroid



def get_segmentation_coordinates(segmented_image):
	"""
	segmented_image: numpy array (height, width, 1) where anything above 0 is foreground
	particle_centroids: returns the center of each discrete segmented particle as a list.
	"""

	# Convert predicted image to binary (in case there are multiple classes)
	pred_binary = (segmented_image[:,:] > 0).astype('uint8')

	# Erosions
	# Note: Erosions allows for 1) seperation of merged blobs and 2) removal of popcorn prediction noise. 
	struct_element = np.ones((2,2), np.uint8)
	pred_binary = cv2.erode(pred_binary.astype('float32'), struct_element, iterations=1)

	# Input requirement: 8-bit single, channel image. Image input is binary with all non-zero pixels treated as 1s. 
	# connected_output array: [num_labels, label_matrices, marker_stats, centroids]
	# Connectivity: Connectivity of 8 makes it more likely that particles that are merged due to proximity will be treated seperately. 
	pred_connected = cv2.connectedComponentsWithStats(pred_binary.astype('int8'), connectivity=8)
	particle_centroids = pred_connected[3][1:] # Remove the background centroid (at index 0).


	return particle_centroids


def get_scale_factors(original_image, resized_image): 
	"""
	Returns the scale factor, so that resized_image_size = scale_factor*original_image
	Args
	original_image: Numpy array of orginal image
	resized_image: Numpy array of resized image
	"""
	original_height = original_image.shape[0]
	original_width = original_image.shape[1]
	resized_height = resized_image.shape[0]
	resized_width = resized_image.shape[1]
	factor_height =  resized_height/float(original_height)
	factor_width = resized_width/float(original_width)

	return factor_height, factor_width	



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


def initialize_segmentation_model():
	""" 
	Builds the semantic segmentation model and loads the pre-trained weights. 
	"""
	# Instantiates configuration for training/validation
	config = SegmentParticles_Config()

	# Configuration sanity check
	CNN_functions.validate_segmentation_config(config)

	# Instantiate training/validation data
	data = SegmentParticlesData(config)

	# Print configuration
	CNN_functions.print_configurations(config) # Print config summary to log file
	data.print_data_summary() # Print data summary to log file

	# Builds model
	model = createModel(input_shape = config.image_shape, base_weights = config.imagenet_weights_file, classes=config.nclasses)

	# Load weights (if the load file exists)
	CNN_functions.load_model(model, config.weight_file_input, config)

	return model, data

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