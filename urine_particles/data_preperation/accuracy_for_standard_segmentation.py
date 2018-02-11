# Import basic libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

# import from local libraries
import utility_functions_data as util
sys.path.insert(0, './urine_particles')
import CNN_functions
from SegmentParticles_config import SegmentParticles_Config

""" Configuration """
# Files/Folders
class_name = '10um'
img_filename = '10um_img1.bmp'
mask_filename = '10um_img1.bmp'
img_input_path = './urine_particles/data/clinical_experiment/image_data/20180120_training/segmentation/val_images/' + class_name + '/' + img_filename
mask_input_path = './urine_particles/data/clinical_experiment/image_data/20180120_training/segmentation/val_annotations/' + class_name + '/' + mask_filename
output_folder = './urine_particles/data/clinical_experiment/image_data/20180120_training/segmentation/img_output/'

min_particle_size = 3





def main():
	# Load Image: Ensure image is in grayscale. 
	im_original = cv2.imread(img_input_path, cv2.IMREAD_GRAYSCALE)
	im_mask = cv2.imread(mask_input_path, cv2.IMREAD_GRAYSCALE)

	# Get segment configuration data. Needed for log, etc. 
	config = SegmentParticles_Config()


	# Apply standard segmentation algo
	img_segmented_pred = CNN_functions.standard_segmentation(im_original, min_particle_size)

	# Calculate accuracy. 
	output_filename_suffix = img_filename[:img_filename.rfind('.')]
	save_path = output_folder + output_filename_suffix
	CNN_functions.determine_segmentation_accuracy(
		truth_matrix=im_mask, 
		pred_matrix=img_segmented_pred, 
		config=config, 
		radius=config.detection_radius, 
		base_output_path=save_path)



if __name__ == "__main__":
	main()