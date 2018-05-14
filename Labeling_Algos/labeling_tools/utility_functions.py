import os
import cv2



def crop_images(log_path, output_folder, crop_increment, crop_size):
	"""
	Crops both the original image and the annotations into the output_folder. 
	"""

	# Define directory/file-names
	image_file_name, subfolder = get_image_file_name(log_path)
	main_class = subfolder.split('/')[-3]
	input_image_path = subfolder + "original/" + image_file_name + ".bmp"
	input_annotations_path = subfolder + "annotations_output/" + output_folder +  image_file_name + ".bmp"
	output_image_dir = subfolder + "original/" + output_folder + "crops_output/"
	output_annotations_dir = subfolder + "annotations_output/" + output_folder + "crops_output/"

	# Make necessary directories
	if (not os.path.isdir(output_image_dir)):
		os.makedirs(output_image_dir)
	if (not os.path.isdir(output_annotations_dir)):
		os.makedirs(output_annotations_dir)


	im_original = cv2.imread(input_image_path) # Careful: format in BGR
	im_mask = cv2.imread(input_annotations_path) # Careful: format in BGR

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
			crop_image_path = output_image_dir + main_class + "_" + image_file_name+"_"+ str(m) + "_" + str(n) + "_cut.bmp"
			crop_annotations_path = output_annotations_dir + main_class + "_" + image_file_name+"_"+ str(m) + "_" +str(n) + "_cut.bmp"

			cv2.imwrite(crop_image_path, im_original[x1:x2, y1:y2,:])
			cv2.imwrite(crop_annotations_path, im_mask[x1:x2, y1:y2,:])


def get_image_file_name(log_path):
	"""
	Return the image_file_name and the subfolder path from the log path.
	Note: Assumes standard subfolder structure. 
	"""
	fileName_start = log_path.rfind("/")
	fileName_end = log_path.rfind("_")
	image_file_name = log_path[fileName_start+1:fileName_end]
	subfolder_end = log_path.rfind("/",0, fileName_start) + 1
	subfolder = log_path[:subfolder_end]
	return image_file_name, subfolder