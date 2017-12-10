# Base packages 
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import cv2
from glob import glob

# Labeling Configuration
root_dir = "./data/20171027/wbc/selected1/"
coordinates_folder = root_dir + "coordinates/"
input_image_dir = root_dir + "original/"
output_overlay_dir = root_dir + "overlay_output/"
output_annotations_dir = root_dir + "annotations_output/"
indicator_radius = 10

# Cropping Configuring
crop_images_flag = True
crop_increment = 420
crop_size = 480


def main(): 

	log_files = glob(coordinates_folder + "*")
	for log_path in log_files:
		# Label Images
		label_image(log_path)
		# Crop Images
		if (crop_images_flag):
			crop_images(log_path)


def label_image(log_path):
	# Define directory/file names
	image_file_name = get_image_file_name(log_path)

	input_image_path = input_image_dir + image_file_name + ".bmp"
	output_image_annotated_path = output_annotations_dir + image_file_name + ".bmp"
	output_image_overlay_path = output_overlay_dir + image_file_name + ".bmp"

	# Load image
	im_overlay = cv2.imread(input_image_path) # Careful: format in BGR
	im_mask = np.zeros(im_overlay.shape)

	# Load the coordinate data structure saved in JSON format. 
	log = open(log_path, 'r')
	particle_list = json.load(log)
	log.close()

	# Label foreground pixels
	for index in range(len(particle_list)):

		centroid = particle_list[index]
		centroid = (int(centroid[0]), int(centroid[1]))
		cv2.circle(im_overlay, center=centroid, radius=indicator_radius, color=(0,0,0), thickness=-1) # Label pixels on overlay image
		cv2.circle(im_mask, center=centroid, radius=indicator_radius, color=(255,255,255), thickness=-1) # Label pixels on mask image

	# Save input image with selected particles shown
	cv2.imwrite(output_image_overlay_path, im_overlay)
	cv2.imwrite(output_image_annotated_path, im_mask)


def crop_images(log_path):

	#Define directory/file-names
	image_file_name = get_image_file_name(log_path)
	input_image_path = input_image_dir + image_file_name + ".bmp"
	output_image_overlay_path = output_annotations_dir + image_file_name + ".bmp"

	im_original = cv2.imread(input_image_path) # Careful: format in BGR
	im_mask = cv2.imread(output_image_overlay_path) # Careful: format in BGR

	vertical_range = int((im_original.shape[0]-crop_size)/float(crop_increment))
	horizontal_range = int((im_original.shape[1]-crop_size)/float(crop_increment))
	for m in range(vertical_range): 
		for n in range(horizontal_range):

			# Select cropping parameters
			x1 = m*crop_increment
			x2 = x1+crop_size
			y1 = n*crop_increment
			y2 = y1+crop_size

			# Save cropped images
			crop_image_path = input_image_dir+"crops_output/"+image_file_name+"_"+ str(m) + "_" + str(n) + "_cut.bmp"
			crop_annotations_path = output_annotations_dir+"crops_output/"+image_file_name+"_"+ str(m) + "_" +str(n) + "_cut.bmp"

			cv2.imwrite(crop_image_path, im_original[x1:x2, y1:y2,:])
			cv2.imwrite(crop_annotations_path, im_mask[x1:x2, y1:y2,:])



def get_image_file_name(log_path):
	fileName_start = log_path.rfind("/")
	fileName_end = log_path.rfind("_coordinates.json")
	image_file_name = log_path[fileName_start+1:fileName_end]
	return image_file_name

if __name__ == "__main__":
    main()
