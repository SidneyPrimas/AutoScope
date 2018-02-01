# Import basic libraries
import os
import numpy as np
import cv2
import glob
import math
from datetime import datetime
import re

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
# Files/Folders
# root_folder contains input_folders (as well as )
root_folder = "./urine_particles/data/clinical_experiment/prediction_folder/sol1_rev1/"
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


def main():

	# Builds the classification model
	model, data = initialize_classification_model()

	all_label_list = []
	for index_folder, target_folder in enumerate(input_folders):

		# Define paths
		image_data_path = root_folder + target_folder + image_data_folder
		sorted_output_path =  root_folder + target_folder + sorted_output_folder
		debug_output_path = root_folder + target_folder + 'debug_output/'

		# Create necessary data generator
		pred_generator = data.create_custom_prediction_generator(pred_dir_path=image_data_path)

		# Calculate number of batches to run. 
		# Batch count selected so we process until we cannot make a full batch with new images. 
		total_cropped_images = len(glob.glob(image_data_path + "images/*.bmp"))
		num_batches = int(math.floor(total_cropped_images/float(data.config.batch_size)))

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
		split_batch_into_class_folders(all_pred, all_path_list, sorted_output_path, class_mapping)

		
		original_img_dic = label_canvas_based_on_crop_filename(label_list, all_path_list, root_folder, data.config.colors)
		# Save the labeled iamges
		for img_name in original_img_dic:
			cv2.imwrite(debug_output_path + img_name + '_labeled_from_crops.bmp', original_img_dic[img_name])


	# Print out results for all folders in input_folders
	data.config.logger.info("Results for: %s", root_folder)
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

def initialize_classification_model():
	""" 
	Builds the classification  model and loads the pre-trained weights. 
	"""

	# Instantiates configuration for training/validation
	config = ClassifyParticles_Config()

	# Instantiate training/validation data
	data = ClassifyParticlesData(config)

	# Builds model
	model = createModel(input_shape = config.image_shape, base_weights = config.imagenet_weights_file, classes=config.nclasses)


	# Load weights (if the load file exists)
	CNN_functions.load_model(model, config.weight_file_input, config)

	return model, data



if __name__ == "__main__":
	main()