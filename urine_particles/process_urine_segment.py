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
from SegmentParticlesData import SegmentParticlesData
import CNN_functions
from segmentation_models import FCN8_32px_factor as createModel
from SegmentParticles_config import SegmentParticles_Config

"""
Description: 

Execution Notes: 


To Do: 
+ Need to consolidate create_classification_folder_from_labels.py, create_classification_folder_from_model.py and segment_urine.py
++ These files are based on each other and they have A LOT of overlapping code. 
+ Saving results: In multiple functions I save the results of the function (instead of or in addition to returning them)
++ For a more modular implementation, return the results and then have the user of the function save the results. 
"""

""" Configuration """
# Files/Folders
root_folder = "./urine_particles/data/clinical_experiment/prediction_folder/sol1_rev1/" # Folder that contains files to be processes
input_files = ["img1.bmp", "img2.bmp", "img3.bmp", "img4.bmp", "img5.bmp", "img6.bmp"] # Name of files to be processed. 
output_folders = ["cropped_output/", "cropped_output/", "cropped_output/", "cropped_output/", "cropped_output/", "cropped_output/"] # Name of the output folder. 

output_crop_size = 64 # The output size of the crops, measured in pixels. Used on the original image. 
indicator_radius = 32

debug_flag = True
keep_boundary_particles = False



def main():

	# Clean up disk from previous sessions
	clean_up_old_output_folders(root_folder, output_folders)


	# Build the semantic segmentation model. 
	model, data = initialize_segmentation_model()


	for file_index, target_file in enumerate(input_files):

		# Define files/folders
		target_file_path = root_folder + target_file
		output_folder_path = root_folder + output_folders[file_index]

		# Build output folders (if necessary)
		build_segmentation_output_folder(output_folder_path)

		# Generate centroid list based on segmentation model
		centroid_list = get_centroid_list(model, data, target_file_path, output_folder_path)


		# Generate crops on input image based on the centroid_list (produced by segmentation model)
		crop_based_on_centroids(target_file_path, centroid_list, output_folder_path)





def get_centroid_list(model, data, target_file_path, output_folder_path):
	"""
	Description: Using the segmentation model, get the list of centroids generated from the target_file_path. 
	# Possible To Do: Move saving results out of this function (should be done by user of function)
	model: Segmentation model
	data: object used to perform segmentations
	target_file_path: The path to the input image to be segmented. 
	output_folder_path: Location where outputs will be saved (centroid list and segmented image)
	"""

	# Get output prefix name
	output_file_prefix = get_file_name(target_file_path, remove_ext=True)

	# Predict segmentation of original_img with segmentation model  
	pred_array = data.predict_image(model, target_file_path)

	# Get particle list from prediction 
	pred_image = CNN_functions.preprocess_segmentation(pred_array, target_size=data.config.target_size, apply_morphs=True)

	# Obtain centroid list with connected component analysis
	pred_connected = cv2.connectedComponentsWithStats(pred_image.astype('int8'), connectivity=8)
	centroid_list = np.asarray(pred_connected[3][1:])  # Remove the background centroid (at index 0).


	# Save results: Segmented image
	output_debug_path_prefix = output_folder_path + "debug_output/" + output_file_prefix 
	img_output = CNN_functions.get_color_image(pred_image, data.config.nclasses, data.config.colors)
	cv2.imwrite(output_debug_path_prefix + "_segmented.bmp", img_output)

	# Save results: centroid list
	output_log = open(output_debug_path_prefix + "_centroid_list.log" , 'w+')
	json.dump({"centroid_list":centroid_list.tolist()}, output_log)
	output_log.close()


	return centroid_list




def crop_based_on_centroids(target_file_path, centroid_list, output_folder_path):
	"""
	Description: Crops the input image based on the centroids provided. 
	Args:
	centroid_list: List of centroids produced by connectedComponentsWithStats
	output_folder_path: The path were the cropped images will be stored (as well as other outputs)
	"""

	### Setup ###
	# Load image
	original_img = cv2.imread(target_file_path) 
	original_cpy = None
	if (debug_flag):
		original_cpy = original_img.copy()

	# Get output prefix name
	output_file_prefix = get_file_name(target_file_path, remove_ext=True)


	# Crop each labeled particle from the original image. 
	centroid_list = np.asarray(centroid_list)
	for index, centroid in enumerate(centroid_list): 

		# Crop Original image
		x1, x2, y1, y2 = CNN_functions.get_crop_coordinates(original_img.shape, centroid, output_crop_size)
		crop = original_img[y1:y2,x1:x2]


		# Save cropped images
		# Crop particles when 1) the particle has the correct dimensions (not a boundary particle) or when we crop all particles (even boundary particles)
		crop_has_target_dim = (crop.shape[0] == output_crop_size) and (crop.shape[1] == output_crop_size)
		if crop_has_target_dim or keep_boundary_particles:
			# Save cropped image
			file_name =  output_file_prefix + "_" + str(int(centroid[0])) + "_" + str(int(centroid[1])) + ".bmp"
			output_path = output_folder_path + "data/images/" + file_name
			cv2.imwrite(output_path, crop)
			indicator_color = (0, 255, 0) # Indicator color when cropped
		else:
			indicator_color = (255, 0, 0) # Indicator color when not cropped
		

		# Place indicator for each ground truth partidcle detected. 
		if (debug_flag): 
			circl_centroid = (int(centroid[0]), int(centroid[1]))
			cv2.circle(
				original_cpy, 
				center=circl_centroid, 
				radius=indicator_radius, 
				color=indicator_color, 
				thickness=2)

			
	# Save output with cropped images
	if (debug_flag):
		debug_img_output_path = output_folder_path + "debug_output/" + output_file_prefix + "_flagged_crops.bmp"
		cv2.imwrite(debug_img_output_path, original_cpy)
	

def scale_centroid(centroid, factor):
	"""
	Scale the x/y of the centroid by factor. 
	centroid: list/tuple of centroid [x, y]
	factor:list/tuple of factor [x_factor, y_factor]
	scaled_centroid: the scaled centroid [x*x_factor, y*y_factor]
	"""
	scaled_centroid = [centroid[0]*factor[0], centroid[1]*factor[1]]
	return scaled_centroid


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

	# Builds model
	model = createModel(input_shape = config.image_shape, base_weights = config.imagenet_weights_file, classes=config.nclasses)

	# Load weights (if the load file exists)
	CNN_functions.load_model(model, config.weight_file_input, config)

	return model, data


def build_segmentation_output_folder(output_folder_path):

	if (not os.path.isdir(output_folder_path)): 
		os.makedirs(output_folder_path)
		os.makedirs(output_folder_path + "data/images/")
		os.makedirs(output_folder_path + "debug_output/")

def clean_up_old_output_folders(root_folder, output_folders):

	for target_folder in output_folders:

		# Define files/folders
		output_folder_path = root_folder + target_folder

		# Remove folder if it exists
		CNN_functions.delete_folder_with_confirmation(output_folder_path)



def get_file_name(target_file_path, remove_ext=True):
	"""
	Description: From a path to a file, returns the file name. 
	remove_ext: Indicate if the extension should be removed.
	"""
	file_name_end = -1
	if remove_ext:
		file_name_end = target_file_path.rfind(".")
	file_name_start = target_file_path.rfind("/") + 1
	output_file_prefix = target_file_path[file_name_start:file_name_end]

	return output_file_prefix



if __name__ == "__main__":
	main()