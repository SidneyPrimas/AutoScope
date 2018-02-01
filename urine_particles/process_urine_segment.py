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
# Folder that contains the canvas files to be processed
root_folder = "./urine_particles/data/clinical_experiment/prediction_folder/sol1_rev1/" 
 # Name of files to be processed. 
input_files = ["img1.bmp", "img2.bmp", "img3.bmp", "img4.bmp", "img5.bmp", "img6.bmp", "img7.bmp"]
 # Name of the output folder (placed in the root folder)
output_folders = ["img1/", "img2/",  "img3/", "img4/", "img5/", "img6.bmp/", "img7.bmp/"]


# Select 'crops' to produce crops from the segmentation. Select 'semantic' get particle statistics for images based on semantic segmentation.
segmentation_mode = 'semantic' 
class_mapping =  {0:'back', 1:'10um', 2:'rbc', 3:'wbc'}
# The label of the class to be discarded. This usually is the lave of 'other' or 'background'
discard_label = 0 
# The output size of the crops, measured in pixels. Used on the original image. 
output_crop_size = 64 
indicator_radius = 32

# Flags
debug_flag = True
keep_boundary_particles = False




def main():

	# Clean up disk from previous sessions
	clean_up_old_output_folders(root_folder, output_folders)

	# Build the semantic segmentation model. 
	model, data = initialize_segmentation_model()

	# Process images in either 'crops' or 'semantic' mode
	if (segmentation_mode == 'crops'): # Produce crops from segmentation
		process_inputImages_in_crops_mode(model, data, root_folder, input_files, output_folders)
	elif (segmentation_mode == 'semantic'): # Use semantic segmentation to indicate particles
		process_inputImages_in_semantic_mode(model, data, root_folder, input_files, output_folders)
	else: 
		raise RuntimeError("The segmentation mode that was selected doesn't exist. Please select a correct mode.")



def process_inputImages_in_crops_mode(model, data,  root_folder, input_files, output_folders):
	"""
	Description: Process all images by generating crops around the predicted particle location (predicted through segmentation)
	"""
	for file_index, target_file in enumerate(input_files):

		# Define files/folders
		target_file_path = root_folder + target_file
		output_folder_path = root_folder + output_folders[file_index]

		# Build output folders (if necessary)
		build_segmentation_output_folder(output_folder_path)

		# Produce crops 
		generate_crops_from_inputImage(model, data, target_file_path, output_folder_path)


def process_inputImages_in_semantic_mode(model, data,  root_folder, input_files, output_folders):
	"""
	Description: Process all images by counting particles through semantic segmentation.
	Print out average results across all images to log. 
	"""
	all_labels_list = []
	for file_index, target_file in enumerate(input_files):

		# Define files/folders
		target_file_path = root_folder + target_file
		output_folder_path = root_folder + output_folders[file_index]

		# Build output folders (if necessary)
		build_segmentation_output_folder(output_folder_path)

		# Produce labels from semantic segmentation of a single canvas image
		label_list = generate_particlePredictions_from_inputImage(model, data, target_file_path, output_folder_path)
		all_labels_list.extend(label_list)

	canvas_img_cnt = len(input_files)
	data.config.logger.info("\nResults for: %s", root_folder)
	CNN_functions.print_summary_statistics_for_labels(all_labels_list, class_mapping, data.config, discard_label=discard_label, image_count=canvas_img_cnt)



def generate_particlePredictions_from_inputImage(model, data,  target_file_path, output_folder_path): 
	"""
	Description: Use semantic segmetnation approach to classify particles on a SINGLE image. 
	"""

	# Get output prefix name
	output_file_prefix = get_file_name(target_file_path, remove_ext=True)

	# Predict segmentation of original_img with segmentation model  
	pred_array = data.predict_image(model, target_file_path)

	# Obtain original image
	original_img = cv2.imread(target_file_path)

	# Temp (use for debugging as to generate predictions more rapidly)
	#temp_prediction_path = root_folder + output_file_prefix + "_rawImgArray.bmp"
	#pred_array = cv2.imread(temp_prediction_path, cv2.IMREAD_GRAYSCALE)

	# Convert to labeled image.
	pred_img = CNN_functions.predArray_to_predMatrix(pred_array, data.config.target_size)

	# Apply morphological operations
	morph_mask = CNN_functions.apply_morph(pred_img, morph_type='classes')

	# From the semantic image, get list of partiles with the corresponding classes
	centroid_list, component_labels, particle_label_mask = semanticImg_to_particleCounts(pred_img, morph_mask)

	# Place indicator for each classified particle. 
	# TODO: Break out into seperate function for readability (maybe?)
	for index, centroid in enumerate(centroid_list):
		label = component_labels[index]
		circl_centroid = (int(centroid[0]), int(centroid[1]))
		cv2.circle(
			original_img, 
			center=circl_centroid, 
			radius=indicator_radius, 
			color=data.config.colors[label], 
			thickness=2)

	# Convert labeled images to rgb images
	# Produce rgb image where each connected component is labeled with it's label
	particle_label_img_rgb = CNN_functions.get_color_image(particle_label_mask, nclasses=data.config.nclasses, colors=data.config.colors)
	# Produce rgb image for predicted image from semantic segmentation
	pred_img_rgb = CNN_functions.get_color_image(pred_img, nclasses=data.config.nclasses, colors=data.config.colors)
	
	
	# Produce summary statistics
	data.config.logger.info("\nResults for: %s", target_file_path)
	CNN_functions.print_summary_statistics_for_labels(component_labels, class_mapping, data.config, discard_label=discard_label, image_count=1)


	# Save images
	debug_images_path = output_folder_path + "debug_output/" + output_file_prefix
	cv2.imwrite(debug_images_path + "_model_raw_prediction.bmp" , pred_img_rgb)
	cv2.imwrite(debug_images_path + "_processed_prediction.bmp" , particle_label_img_rgb)
	cv2.imwrite(debug_images_path + "_labeled_form_semantic.bmp" , original_img)
	

	return component_labels




def semanticImg_to_particleCounts(semantic_img, mask):
	"""
	Description: Determines particle counts by looking at each connected component in mask and counting the underlying labels in semantic_img that make up each connected component. 
	Args
	semantic_img: Image with each pixel labeled with a class. 
	mask: Binary image to identify location of particles through connected component analysis. 
	Returns
	centroid_list: List of particle centroids, without the background
	component_labels: List of particle labels, withouththe background
	particle_label_mask: Image where connected component particles are labeled with the final class. 
	"""
	# Seperate connected components based in input mask
	connectedComponents_output = cv2.connectedComponentsWithStats(mask.astype('int8'), connectivity=8)
	connectedComponents_num = connectedComponents_output[0]
	connectedComponents_mask = connectedComponents_output[1] 
	centroid_list = connectedComponents_output[3][1:] # Remove background centroid

	

	# Classify each connected component based on the labels within the semantic_img
	particle_label_mask = np.zeros(semantic_img.shape)
	component_labels = [] # list of all the images labels
	for component_index in range(connectedComponents_num):

		#  The first label is the background (zero label). We always ignore it. 
		if component_index == 0:
			continue

		# Creates mask for a single connected component
		target_component_mask = (connectedComponents_mask == component_index)
		# Gets the labels in semantic_img that correspond to a single connected component. 
		target_component_labels = semantic_img[target_component_mask]
		# Calculates the particles single lable given a list of the labels for each pixel that particle has. 
		particle_label = get_particleLabel_from_pixelLabels(target_component_labels, background=discard_label)
		# Add label to list of labels for image
		component_labels.append(particle_label)
		
		# Generate output image
		particle_label_mask += target_component_mask * particle_label 


	return centroid_list, component_labels, particle_label_mask


def get_particleLabel_from_pixelLabels(pixel_labels, background = 0):
	"""
	Description: A particle is made from many pixel labels. Given a list of pixel labels, find the label with the most pixels. 
	Discard the background label as irrelevant. 
	pixel_labels: A list of labels. 
	background: The background label should never be returned. It should be discarded. 
	"""

	# Get counts for each label
	unique_labels, counts = np.unique(pixel_labels, return_counts=True) 
	# Transform results into a dictionary
	counts_dict = dict(zip(unique_labels, counts))
	# Sort dictionary based on value
	ordered_labels = sorted(counts_dict.items(), key=lambda x:x[1], reverse=True) 
	# Identify label with most counts. Disregard counts of background label.
	label_with_max_counts = ordered_labels[0][0]
	if (label_with_max_counts == background):
		label_with_max_counts = ordered_labels[1][0]

	return label_with_max_counts
	


def generate_crops_from_inputImage(model, data, target_file_path, output_folder_path):
	"""
	Description: Use connectedComponents to determine predicted centroids of particles.
	Create crops around each centroid to be classified with another model. Do for single image. 
	"""

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

	# Apply morphological transformations
	pred_image = CNN_functions.predArray_to_predMatrix(pred_array, data.config.target_size) # Convert from prediction arrays to labeled matrices
	pred_image = CNN_functions.apply_morph(pred_image, morph_type='foreground')

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
