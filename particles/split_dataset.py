"""
    File name: split_dataset.py
    Author: Sidney Primas
    Date created: 4/17/2017
    Python Version: 2.7
    Description: Split the dataset into the training images and the validation images.  
    Each folder in the source directry can only have images (and no sub-folders). 
"""

import numpy as np
import math
from glob import glob
import os


### Configurable Settings ###
debug_flag = 1
validation_percent = 0.1
source_data_directory = "./data/IrisDB_resampling/"

# Obtain images from all classes
directory_list = glob(source_data_directory + "*")

# Create the training and validation folders in the source data directory
os.mkdir(source_data_directory + "Validation")
os.mkdir(source_data_directory + "Training")

# Iterate through each class (which is a directory in the source data directory)
for directory in directory_list:
	# Get the name of the directory
	start = directory.rfind("/")
	directory_name = directory[start+1:]

	# Create the directory in the validation and training folder 
	os.mkdir(source_data_directory + "Validation/" + directory_name)
	os.mkdir(source_data_directory + "Training/" + directory_name)

	# Get all the  images int he directory
	image_list = glob(directory + "/*.jpg")
	validation_size = math.ceil(len(image_list)*validation_percent)

	# Review of images in directory
	if debug_flag:
		print "Directory Path: %s" % directory
		print "Directory Name: %s" % directory_name
		print "Total Images: %d" % len(image_list)
		print "Validation Size: %d" % validation_size
		print "Training Size: %d \n" % (len(image_list)-validation_size)

	# Sort the images into validation and training folders
	for image_path in image_list:

		# Get the image number 
		start = image_path.rfind("/")
		end = image_path.rfind(".jpg")
		image_num = int(image_path[start+1:end])

		# Move images to VALIDATION Folder
		if image_num < validation_size:
			destination_for_file = (source_data_directory + "Validation/" + directory_name + "/" + str(image_num) + ".jpg")
			os.rename(image_path, destination_for_file)
		
		# Move images to TRAINING Folder	
		else:
			destination_for_file = (source_data_directory + "Training/" + directory_name + "/" + str(image_num) + ".jpg")
			os.rename(image_path, destination_for_file)


