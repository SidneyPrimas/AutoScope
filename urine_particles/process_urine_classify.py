# Import basic libraries
import os
import numpy as np
import cv2
from glob import glob
import math
from datetime import datetime
import re
import argparse

#Import keras libraries
from tensorflow.python.keras.optimizers import SGD, RMSprop
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.utils import plot_model

# import from local libraries
from ClassifyParticlesData import ClassifyParticlesData
import CNN_functions
from classification_models import base_model as createModel
from ClassifyParticles_config import ClassifyParticles_Config

"""
Description: Given a setup of croped particles, classify particles. Provide statistics on predictions. 
"""

""" Configuration """
# root_folder contains input_folders (as well as )
root_folder = "./urine_particles/data/clinical_experiment/prediction_folder/sol1_rev1/"
# If True, auto selectes all '.bmp' images in root folder. 
auto_determine_inputs = True

# Files/Folders
# input_folder is a set of folders (with specific sub-folders). One of these subfolders includes the crops. 
# All files related to a classification session are loaded and saved into a single input_folder
input_folders = ["img1/", "img2/",  "img3/", "img4/", "img5/", "img6/", "img7/"]
# image_data_folder contains subfolders that contain the crops to be predicted
image_data_folder =  "data/"
# sorted_output stores references to the sorted images for user review. 
sorted_output_folder = "sorted_output/"

# The numbe of canvas images that went into each of the input folders. 
input_img_count = [1, 1, 1, 1, 1, 1, 1]
class_mapping =  {0:'10um', 1:'other', 2:'rbc', 3:'wbc'}
discard_label = 1
indicator_radius = 32

# Flags
split_crops_into_class_folders_flag = False


def main():

	# Determine files to be processed automatically. Custom function depending on folder setup and training goals.
	# Allows for more rapid cloud processing. Function updated often. 
	# Current version: Assumes assumes single input_folder per original image. 
	if (auto_determine_inputs):
		root_folder, input_folders, input_img_count = auto_determine_classification_config_parameters(output_folder_suffix="crops")

	# Builds the classification model
	model, data = initialize_classification_model(log_dir=root_folder)

	# Print Configuration
	data.config.logger.info("Segmentation Prediction Results")
	data.config.logger.info(root_folder)
	data.config.logger.info(input_folders)
	data.config.logger.info(input_img_count)


	all_label_list = []
	for index_folder, target_folder in enumerate(input_folders):

		# Indicate next target_folder in log
		data.config.logger.info("Results for: %s", root_folder)

		# Define paths
		image_data_path = root_folder + target_folder + image_data_folder
		sorted_output_path =  root_folder + target_folder + sorted_output_folder
		debug_output_path = root_folder + target_folder + 'debug_output/'


		# Create necessary data generator
		pred_generator = data.create_custom_prediction_generator(pred_dir_path=image_data_path)

		# Calculate number of batches to run. 
		# When round number of batches DOWN: 
		# + Drawback: Certain segmented particles will not be classified. So, results/vizualizations will  not include these.
		# When round number of batches UP: 
		# + Will label all segmented particles. However 1) will have duplicates in labeling (should be fine) and 2) duplicate in statistics.
		# Note: With either approach, the statistics will be randomly biased. With round UP will have all particles included on output images.
		total_cropped_images = len(glob(image_data_path + "images/*.bmp"))
		num_batches = int(math.ceil(total_cropped_images/float(data.config.batch_size)))
		data.config.logger.info("Images classified twice: %d", num_batches*data.config.batch_size-total_cropped_images)

		# Predict: Sort crops into classes
		all_pred, all_path_list = data.predict_particle_images(
			model=model, 
			pred_generator=pred_generator, 
			total_batches=num_batches) 

		# Print out results for single folder
		label_list = np.argmax(all_pred, axis=1)
		all_label_list.extend(label_list)
		data.config.logger.info("Results for: %s", image_data_path)
		CNN_functions.print_summary_statistics_for_labels(
			label_list, 
			class_mapping, 
			data.config, 
			discard_label= discard_label, 
			image_count = input_img_count[index_folder])


		# Sort images into class folders. 
		if (split_crops_into_class_folders_flag):
			split_batch_into_class_folders(all_pred, all_path_list, sorted_output_path, class_mapping)

		
		original_img_dic = label_canvas_based_on_crop_filename(label_list, all_path_list, root_folder, data.config.colors)
		# Save the labeled iamges
		for img_name in original_img_dic:
			prefix_rootdir = root_folder.split('/')[-2]
			cv2.imwrite(debug_output_path + prefix_rootdir + "_" + img_name + "_labeled_from_crops.bmp", original_img_dic[img_name])


	# Print out results for all folders in input_folders
	total_input_images = sum(input_img_count)
	CNN_functions.print_summary_statistics_for_labels(
		all_label_list, 
		class_mapping, 
		data.config, 
		discard_label= discard_label, 
		image_count = total_input_images)


def label_canvas_based_on_crop_filename(label_list, all_path_list, root_folder, label_colors):
	"""
	Description: Label predicted particles on canvas image (original image).
	Use information in the cropped particles path to 1) determine centroid and 2) determine which original image to label. 
	Args
	label_list: List of labels. Labels are in same order as paths in all_path_list
	all_path_list: List of paths, including filename of the crops
	root_folder: The folder containing the original images. 
	label_colors: RGB colors of size nclasses. 
	Return 
	original_img_dic: Dictionary that maps the base name (no extension) of the original image to the labeled original image
	"""
	original_img_dic = {}
	for index_path, crop_path in enumerate(all_path_list): 

		# Identify crop_path file name
		crop_filename = crop_path[crop_path.rfind('/')+1:]

		# Identify original image name
		original_img_name = crop_filename[:crop_filename.find('_')]
		

		# Get reference to original image: either load it (and store in dic for future reference) 
		# Or get referene from original_img_dic
		if original_img_name in original_img_dic:
			original_img = original_img_dic[original_img_name]
		else:
			# Load the original image, and store it in dictionary
			original_img_path = root_folder + original_img_name + ".bmp"
			original_img = cv2.imread(original_img_path)
			original_img_dic[original_img_name] = original_img



		# Identify particle centroid from crop image name
		coordinates_str = re.findall("_(\d+)", crop_filename)
		circle_centroid = (int(coordinates_str[0]), int(coordinates_str[1]))


		# Place indicator for each classified particle. 
		label = label_list[index_path]
		cv2.circle(
			original_img, 
			center=circle_centroid, 
			radius=indicator_radius, 
			color=label_colors[label], 
			thickness=2)


	return original_img_dic





def split_batch_into_class_folders(label_pred, path_list, output_dir, labels_to_class):
	"""
	Description: Takes a batch of images with correspoinding categorigcal/softmax predictions. Uses output_dir and labels_to_class to store outputs. 
	output_dir: Directory into which predicted images will be saved. 
	labels_to_class: Dictionary structure that has labels as keys to the class names. 
	Note: Create a permenant link. to corresponding image. 
	"""

	# Setup output directory		
	build_sorted_output_folder(output_dir, labels_to_class)

	for index in range(label_pred.shape[0]):
		label = np.argmax(label_pred[index])
		datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H-%M-%S-%f')
		file_name = str("%0.04fconfidence_%s"%(label_pred[index][label], datestring))
		img_save_path = output_dir + labels_to_class[label] + "/" + file_name + ".bmp"
		os.link(path_list[index], img_save_path)



def build_sorted_output_folder(output_dir, labels_to_class):
	"""
	Description: Builds a folder in output_dir with the classes in labels_to_class as subdirectories. 
	Note: Seperate function to improve readabiltiy 
	"""
	# Delete the folder if it exists
	CNN_functions.delete_folder_with_confirmation(output_dir)	

	# Make fodler
	os.makedirs(output_dir)
	for label, class_name in labels_to_class.iteritems():
		os.makedirs(output_dir + class_name)

def initialize_classification_model(log_dir=None):
	""" 
	Builds the classification  model and loads the pre-trained weights. 
	"""

	# Instantiates configuration for training/validation
	config = ClassifyParticles_Config()

	# Reroute/update logging
	if (log_dir):
		prefix = log_dir.split('/')[-2]
		log_file_name = prefix + "_classification_prediction.log" #If None, then name based on datetime.
		config.logger = CNN_functions.create_logger(log_dir, file_name=log_file_name, log_name="predict_classify_logger")


	# Print configuration
	CNN_functions.print_configurations(config) # Print config summary to log file


	# Instantiate training/validation data
	data = ClassifyParticlesData(config)

	# Builds model
	model = createModel(input_shape = config.image_shape, base_weights = config.imagenet_weights_file, classes=config.nclasses)


	# Load weights (if the load file exists)
	CNN_functions.load_model(model, config.weight_file_input, config)

	return model, data

def auto_determine_classification_config_parameters(output_folder_suffix):
	"""
	Determine files/outpus to be processed automatically. Custom function depending on folder setup and training goals.
	Allows for more rapid cloud processing. Function updated often. 
	Current version: Assumes assumes single input_folder per original image. 
	"""

	# Get root folder from user
	parser = argparse.ArgumentParser()
	parser.add_argument("-r","--root_folder", help="Path to root folder that contains data to be processed", type=str)
	args = parser.parse_args()
	root_folder = args.root_folder

	# Check user input
	if (root_folder == None):
		raise ValueError("When auto_determine_inputs is set to True, user needs to provide root_folder.")

	if (not os.path.isdir(root_folder)): 
		raise ValueError("When auto_determine_inputs is set to True, user needs to provide valid root folder.")


	input_files_paths =  glob(root_folder + "*.bmp")
	input_files_paths.sort() 
	input_folders = [CNN_functions.get_file_name_from_path(path)+"_"+output_folder_suffix+"/" for path in input_files_paths]
	input_img_count = [1 for _ in input_files_paths]

	return root_folder, input_folders, input_img_count

if __name__ == "__main__":
	main()