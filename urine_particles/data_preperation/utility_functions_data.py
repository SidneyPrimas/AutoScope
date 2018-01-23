import cv2
from glob import glob
import os
import math

def label_single_image(input_image, particle_list, color_list, radius, segmentation_labels):
	"""
	Args
	input_image: numpy input image that will be labeled. 
	particle_list: list that includes values with the following format.  [centroid, metadata]
	color_list: List of colors with a size equal to nclasses. 
	radius: the indicator radius
	Return
	labeled_image: The color channels of the image cotains the color labels according to the information in particle_list.
	Note: Technically, the labeled_image doesn't need to be returned since the input_image is a reference to the object. However, we return labeled_image for clarity. 
	"""

	for index in range(len(particle_list)):

		# Obtain + format centroid
		centroid = particle_list[index][0]
		centroid_cv2 = (int(centroid[0]), int(centroid[1]))

		# Obtain the color
		class_metadata = particle_list[index][1]
		pixel_label = metadata_to_label(class_metadata, segmentation_labels)
		indicator_color = color_list[pixel_label] 


		cv2.circle(input_image, center=centroid_cv2, radius=radius, color=indicator_color, thickness=-1) # Label pixels on overlay image


	return input_image

def get_image_name_from_coordinate_log(log_path):
	"""
	Return the image_file_name and the folder that contains it. 
	Note: Assumes standard  structure. 
	"""
	fileName_start = log_path.rfind('/')
	fileName_end = log_path.rfind('_')
	image_file_name = log_path[fileName_start+1:fileName_end]
	folder_end = log_path.rfind('/',0, fileName_start) 
	folder_start = log_path.rfind('/', 0, folder_end)+1
	folder = log_path[folder_start:folder_end]
	return image_file_name, folder


def metadata_to_label(class_metadata, label_struct): 
	main_class = class_metadata["main"]
	sub_class = class_metadata["sub"]

	label = label_struct[main_class][sub_class]

	return label

def split_data(input_dir, output_dir, move_proportion):
	"""
	For each particle folder, move 'move_proportion' of files to the particle folder in the output directory. 
	Move at least one image from each particle folder. 
	"""


	input_subfolder_path_list = glob(input_dir + "*")

	# Process each particle folder. 
	for input_subfolder_path in input_subfolder_path_list:
		# Identify subfolder name
		subfolder_name = input_subfolder_path.split("/")[-1]
		target_output_dir = output_dir + subfolder_name + "/"

		# Create output folder (if needed)
		if (not os.path.isdir(target_output_dir)): 
			os.makedirs(target_output_dir)

		images_list = glob(input_subfolder_path + "/*.bmp")
		images_list.sort() # Ensures that images will be appropriately sorted. 

		count = len(images_list)
		val_count = int(math.ceil(count * move_proportion))

		assert(count >= 2) # Need at least two images in order to allow split into validation foler. 

		# Move each image to output directory (up to val_count)
		for i in range(val_count):
			source = images_list[i] # Get image path to be moved. 

			# Create destination path
			file_name_end = source.rfind(".bmp")
			file_name_start = source.rfind("/") + 1
			base_file_name = source[file_name_start:file_name_end]

			# Get extension information
			extension = source[file_name_end:]
			assert(extension == ".bmp") # In current implementation, assume all extensions are .bmp. 

			destination = target_output_dir + base_file_name + extension
			os.rename(source, destination)

