# Base packages 
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
from glob import glob
import random
import os
import math

# Local libraries
import utility_functions_data as util

"""
Execution Notes: 
+ The raw_image_data folder needs to be pre-populated with data. Specifically, we need a original/coordinates folder for each type of particle.
+ If you want to just label with foreground/background, use class_to_label to indicate foreground/background. 

Implementation Notes: 
+ Take original images and coordinate logs from raw_image_data folder, label the images, and place them into an output training folder with the appropriate folder structure. 
+ Original images in training folder are linked to their corresponding images in the raw_data folder. Do not change/remove raw data. 

"""

""" User Updated Configuration Parameters"""
input_dir_root = './urine_particles/data/clinical_experiment/raw_image_data/'
output_dir_root = './urine_particles/data/clinical_experiment/image_data/20180205_training_plus/'  
segmentation_folder_name = 'segmentation/'

indicator_radius = 20 # Moved from 15 to 20px because moved from 5MPx to 8MPx (scaled radius based on linear increase of width)
crop_images_flag = False
validation_proportion = 0.2


# Implementation Notes: Labels need to start at 0 (with the bacgkround), and increment monotonically by 1. 
# Structure 1) key => main category : {key => sub-category}
# Structure 2) key => folder_name 	: {key => sub-category}
segmentation_labels = {
	"background"	:	{"background": 0} ,# background always has a 0 label
	"10um"			:	{"particle": 1, "other": 0, "discard": 0, "accident": 0}, 
	"rbc"			:	{"particle": 1, "other": 0, "discard": 0, "accident": 0}, 
	"wbc"			:	{"particle": 1, "other": 0, "discard": 0, "accident": 0}, 
}
nclasses = 2 # Total number of classes, including the background class. 
input_particle_folders = [
	"wbc/", 
	"10um/", 
	"rbc/", 
]


""" Crop Parameters """
crop_increment = 120 # How much the crop window is shifted for each crop 
crop_size = 480 # The size of the crop: crop_size x crop_size 


""" Fixed + autocalculated parameters """
output_dir_segmentation = output_dir_root + segmentation_folder_name # Must be a folder that doesn't exist.
train_annotations_dir = output_dir_segmentation + "train_annotations/"
train_images_dir = output_dir_segmentation + "train_images/"
val_annotations_dir = output_dir_segmentation + "val_annotations/"
val_images_dir = output_dir_segmentation + "val_images/"
output_training_dir = output_dir_segmentation + "img_output/"
segmentation_metadata_path = output_dir_root + 'segmentation_metadata.log'
# Preselect random colors (so we get consistentcy through-out images)
visual_colors = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(nclasses)]
label_colors = [(i,i,i) for i in range(nclasses)]



def main(): 

	create_segmentation_folder(output_dir_segmentation) # Build the training folder structure. 

	for target_folder in input_particle_folders:
		coordinate_folder = input_dir_root + target_folder + "coordinates/"
		coordinate_files = glob(coordinate_folder + "*_classes.json")
		for coordinate_log_path in coordinate_files:

			label_classes(coordinate_log_path)


	# Move data from training folder into validation folder. 
	# Since split_data sorts the data, will move corresponding images/annotations to the validation folder. 
	util.split_data(input_dir = train_annotations_dir, output_dir = val_annotations_dir, move_proportion = validation_proportion, in_order = True)
	util.split_data(input_dir = train_images_dir, output_dir = val_images_dir, move_proportion = validation_proportion, in_order = True)

	# Create metadata log
	create_segmentation_metadata_log(segmentation_metadata_path) 




def label_classes(coordinate_log_path, crop_bool = False):
	"""
	Use log file that indicates 1) coordinate of each particle and 2) type of each particle to perform semantic segmentation. 
	"""

	# Define directory/file names
	base_file_name, particle_folder = util.get_image_name_from_coordinate_log(coordinate_log_path)
	output_file_name = particle_folder + "_" + base_file_name 

	# Define output paths
	input_image_path = input_dir_root + particle_folder + "/original/" + base_file_name + ".bmp"
	output_annotated_train_path = train_annotations_dir + particle_folder + "/" + output_file_name + ".bmp"
	output_image_train_path = train_images_dir + particle_folder + "/" + output_file_name + ".bmp"
	output_image_overlay_path = output_training_dir + particle_folder + "/"  + output_file_name + "_overlay_classes.bmp"
	output_image_display_path = output_training_dir + particle_folder + "/"  + output_file_name + "_display_classes.bmp"


	# Load image
	im_original = cv2.imread(input_image_path) # Careful: format in BGR
	im_overlay = im_original.copy()
	im_mask = np.zeros(im_overlay.shape)
	im_mask_display = np.zeros(im_overlay.shape)

	# Load the coordinate data structure saved in JSON format. 
	log = open(coordinate_log_path, 'r')
	coordinate_log_data = json.load(log)
	log.close()

	# Extract json data
	particle_list = coordinate_log_data['particle_list']
	classes = coordinate_log_data['classes']

	verify_class_configuration(classes, segmentation_labels)



	im_overlay = util.label_single_image(im_overlay, particle_list, visual_colors, indicator_radius, segmentation_labels)
	im_mask_display = util.label_single_image(im_mask_display, particle_list, visual_colors, indicator_radius, segmentation_labels)
	im_mask = util.label_single_image(im_mask, particle_list, label_colors, indicator_radius, segmentation_labels)


	# Save the image masks used for visualization. 
	cv2.imwrite(output_image_display_path, im_mask_display)
	cv2.imwrite(output_image_overlay_path, im_overlay)

	# Crops  the annotation image and the original image in the same pattern. 
	if (crop_images_flag):
		crop_image(img = im_mask, output_path = output_annotated_train_path[:-4]) # crop the annotations 
		crop_image(img = im_original, output_path = output_image_train_path[:-4]) # crop the original image 


	else: # Save entire images (annotations/original images) in corresponding training directory.
		cv2.imwrite(output_annotated_train_path, im_mask)
		os.link(input_image_path, output_image_train_path)
	



def crop_image(img, output_path):
	"""
	Crops the img input. Saves the crop in output folder. 
	Args
	img: A numpy image that will be cropped. 
	output_path: The output path of the cropped images, without the file extentsion. 
	"""


	vertical_range = int((img.shape[0]-crop_size)/float(crop_increment))
	horizontal_range = int((img.shape[1]-crop_size)/float(crop_increment))
	for m in range(vertical_range): 
		for n in range(horizontal_range):

			# Select cropping parameters
			x1 = m*crop_increment
			x2 = x1+crop_size
			y1 = n*crop_increment
			y2 = y1+crop_size

			# Save cropped images
			crop_path = output_path + "_" + str(x1) + "_" + str(y1) + ".bmp"

			cv2.imwrite(crop_path, img[x1:x2, y1:y2,:])



def verify_class_configuration(classes_struct, segmentation_labels):
	"""
	Makes sure that segmentation_labels struct contains all the main/sub classes that were used to manually classify the data. 
	"""
	for _, class_metadata in classes_struct.iteritems(): 
		main_class = class_metadata["main"]
		sub_class = class_metadata["sub"]
		if main_class in segmentation_labels:
			if sub_class in segmentation_labels[main_class]:
				continue

		raise UserWarning("Key not in type_to_labe struct.") 



def create_segmentation_folder(path):
	"""
	Create the training folder, including sub-folders.
	"""

	# Need to create a non-existent training folder. Do not delete or over-write an existing folder structure. 
	if (os.path.isdir(path)): 
		raise UserWarning("Training folder already exists. Do not delete or overwrite existing folder structure.")

	# Create the training folder directory structure from scratch. 
	os.makedirs(path) # Create the parent training folder
	create_subfolder_struct(train_annotations_dir)
	create_subfolder_struct(train_images_dir)
	create_subfolder_struct(output_training_dir)

def create_subfolder_struct(path):
	os.mkdir(path)
	for target_folder in input_particle_folders:
		os.mkdir(path + target_folder)


def create_segmentation_metadata_log(path):

	segmentation_metadata = {
		"segmentation_labels": segmentation_labels, 
		"nclasses": nclasses, 
		"input_particle_folders": input_particle_folders, 
		"validation_proportion": validation_proportion, 
		"indicator_radius": indicator_radius
	}


	output_log = open(path , 'w+')
	json.dump(segmentation_metadata, output_log)
	output_log.close()




if __name__ == "__main__":
	main()
