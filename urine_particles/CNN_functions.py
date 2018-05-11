# Import basic libraries
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import json
import os
import cv2
import math
import glob
import shutil
import re

#Import keras libraries
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras._impl.keras import backend as K

# import from local libraries
import ClassifyParticlesData  
import classification_models as createModel
import ClassifyParticles_config


def save_struct_to_file(history, file_path):

	with open(file_path, 'w') as outfile:
	    json.dump(history, outfile)

def print_configurations(config):
	all_class_attributes =  dir(config) #Get all class attributes. 
	local_class_attriubutes = [a for a in all_class_attributes if not a.startswith('__') and not callable(getattr(config,a))]

	config.logger.info("###### CONFIGURATION ######")
	# Print all configuration attributes
	for attribute in local_class_attriubutes: 
		# Since attribute is a string, need to use exec to execute function. 
		exec('config.logger.info("%s: %s"%(attribute, config.' + attribute +'))')
	config.logger.info("Image Data Ordering: %s", K.image_data_format())
	config.logger.info("###### CONFIGURATION ###### \n\n")
		 

def get_json_log(log_path):
	log = open(log_path, 'r')
	json_struct = json.load(log)
	log.close()

	return json_struct

def validate_segmentation_config(config):
	assert(config.nclasses == config.segmentation_metadata["nclasses"]) # Check that the data creation nclasses is the same as the model nclasses. 


def validate_classification_config(config):
	assert(config.nclasses == config.classification_metadata["nclasses"]) # Check that the data creation nclasses is the same as the model nclasses. 
	assert(config.nclasses == len(config.class_mapping))
	print config.nclasses
	print len(glob.glob(config.train_images_dir + "*"))
	assert(config.nclasses == len(glob.glob(config.train_images_dir + "*")))
	assert(config.nclasses == len(glob.glob(config.val_images_dir + "*")))

def create_logger(log_dir, file_name, log_name): 

		logger = logging.getLogger(log_name)
		logger.setLevel(logging.INFO) 
		if (file_name == None):
			file_name = datetime.strftime(datetime.now(), 'log_%Y%m%d_%H-%M-%S.log')

		fileHandler = logging.FileHandler(log_dir + file_name, 'w')
		fileHandler.setLevel(logging.INFO) 
		logger.addHandler(fileHandler)
		consoleHandler = logging.StreamHandler()
		logger.addHandler(consoleHandler)
		#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		#fileHandler.setFormatter(formatter)
		#consoleHandler.setFormatter(logFormatter)
		return logger

def freeze_lower_layers(model, layer_name):
	"""
	Freeze layers below layer_name for Transfer Learning. Need to compile to take into effect. 
	Freeze layer_name and all layers below it. 
	Args:
	model: keras model
	layer_name: Freeze layer_name and all layers below it. 
	"""
	train_this_layer = False
	for layer in model.layers: 
		print layer.name
		layer.trainable = train_this_layer

		# Train layers above layer_name
		if layer.name == layer_name: 
			train_this_layer = True

def get_pixel_accuracy_perBatch(all_truth, all_pred):
	'Return the accuracy given the predicted labels and the ground truth labels for an entire batch.'
	correct_prediction = np.equal(np.argmax(all_truth, axis=2), np.argmax(all_pred, axis=2))
	pixel_accuracy = np.mean(correct_prediction.astype(np.float))

	return pixel_accuracy


def get_pixel_accuracy_perImage(img_truth, img_pred):
	'Return the accuracy given the predicted labels and the ground truth labels for a single image.'
	correct_prediction = np.equal(np.argmax(img_truth, axis=1), np.argmax(img_pred, axis=1))
	pixel_accuracy = np.mean(correct_prediction.astype(np.float))

	return pixel_accuracy


def get_confusion_matrix(all_truth, all_pred): 
	'Caculate a confusion matrix given the predicted labels and the ground truth labels. '

	classes = all_truth.shape[1] # Get total classes
	truth_class = np.argmax(all_truth, axis=1)
	pred_class = np.argmax(all_pred, axis=1) 
	confusion = np.zeros((classes, classes), dtype=float)
	for num, truth_cl in enumerate(truth_class): 
		confusion[truth_cl, pred_class[num]] += 1
	return confusion

def get_classification_accuracy_perBatch(all_truth, all_pred):
	'Return the accuracy given the predicted labels and the ground truth labels. '
	correct_prediction = np.equal(np.argmax(all_truth, axis=1), np.argmax(all_pred, axis=1))
	accuracy = np.mean(correct_prediction.astype(np.float))
	return accuracy


def predArray_to_predMatrix(pred_array, target_size):
	"""
	Description: Convert prediction array (categorical) to a matrix with each pixel labeled with maximum class. 
	"""
	# Convert from categorical format to label format. 
	pred_labeled = np.argmax(pred_array, axis=1) 
	# Reshape into single channel images. 
	pred_matrix = np.reshape(pred_labeled, target_size)
	return pred_matrix



def apply_morph(img_input, morph_type=None):
	"""
	Description: 1) Transforms img_input into binary img, 2) Applies morphological operations to image. 
	Args
	morph_type: 
	+ 'foreground' => applies morph operations for foregound segmentation
	+ 'classes' => applies morph operations for semantic segmentation (per pixel classification) with many classes
	Returns
	img_output: binary img
	"""

	# Argument checking
	if (morph_type != 'foreground') and (morph_type != 'classes') and (morph_type != None):
		raise ValueError("morpth_type in CNN_functions.apply_morph should be either segment or classify")

	# Convert images to binary (if there are multiple classes)
	img_output = ((img_input > 0)).astype('uint8')


	if (morph_type == 'foreground'):
		# Close
		# Note: Closing removes background pixels from foreground blobs. Thus, it consolidates blobs. 
		struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
		img_output = cv2.morphologyEx(img_output.astype('float32'), cv2.MORPH_CLOSE, struct_element, iterations = 2)
		
		# Erosions
		# Note: Erosions allows for 1) seperation of merged blobs and 2) removal of popcorn prediction noise. 
		struct_element = np.ones((4,4), np.uint8)
		img_output = cv2.erode(img_output.astype('float32'), struct_element, iterations=4)


	if (morph_type == 'classes'):
		# Close
		# Note: Closing removes background pixels from foreground blobs. Thus, it consolidates blobs. 
		struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
		img_output = cv2.morphologyEx(img_output.astype('float32'), cv2.MORPH_CLOSE, struct_element, iterations = 2)
		
		# Erosions
		# Note: Erosions allows for 1) seperation of merged blobs and 2) removal of popcorn prediction noise. 
		struct_element = np.ones((4,4), np.uint8)
		img_output = cv2.erode(img_output.astype('float32'), struct_element, iterations=2)

		# Dilate
		# Note: The remaining particle blobs are augmented so that they capture the underlying labels for blob classification. 
		struct_element = np.ones((4,4), np.uint8)
		img_output = cv2.dilate(img_output.astype('float32'), struct_element, iterations=4)


	return img_output

def standard_segmentation(img_input, min_particle_size):
	"""
	Description: Applies standard segmentation algorithm to transform greyscale image to segmented image. 
	Uses adaptive thresholding, morphology functions, and component analysis. 
	Args
	img_input: Raw, grayscale autoscope image. 
	Output: Segmented image, with each pixel labeled as either 0 (background) or 1 (foreground). Single channel image. 
	"""
	
	# Adaptive threshold: Since we have different illuminations across image, we use an adaptive threshold 
	im_thresh = cv2.adaptiveThreshold(img_input, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 4)

	# Closing: When we close an image, we remove background pixel from the foreground. 
	struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
	im_morph = cv2.morphologyEx(im_thresh, cv2.MORPH_CLOSE, struct_element, iterations = 1)


	# Filter markers based on each marker property. 
	connected_output = cv2.connectedComponentsWithStats(im_morph, connectivity=8)
	base_num_labels = connected_output[0]
	print "Number of Original Components: %d" % (base_num_labels - 1) # Minus 1 since we don't count the background
	base_markers = connected_output[1]
	base_stats = connected_output[2]

	# Loop over each marker. Based on stats, decide to eliminate or include marker. 
	for index in range(base_num_labels):
		# The first label is the background (zero label). We always ignore it. . 
		if index == 0:
			continue
		# Area: If any connected component has an area less than "min_particle_size" pixels, turn it into a background (black)
		if base_stats[index][4] <= min_particle_size:
			im_morph[base_markers == index] = 0

	# After the obvious components have been removed, the rest of the components are consolidated by closing the particles. 
	struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
	im_components = cv2.morphologyEx(im_morph, cv2.MORPH_CLOSE, struct_element, iterations = 1)

	# Particles + clumps that are close to each other should be combined. 
	struct_element = np.ones((3,3),np.uint8)
	img_output = cv2.dilate(im_components, struct_element, iterations=5)

	return img_output
	


def get_foreground_accuracy_perImage(truth_array, pred_array, config, radius, base_output_path):
	"""
	Returns the systems accuracy in identifying the location of a particle.
	Preprocesses truth and predicted segmentations arrays from segmentation models (specifically from SegmentParticlesData.py)
	Arguments 
	truth_array: Single ground truth array in (image_pixels, classes)
	pred_array: Single predictions array in (image_pixels, classes)
	"""
	
	# Convert from prediction arrays to labeled matrices
	truth_reshaped = predArray_to_predMatrix(truth_array, config.target_size)
	pred_reshaped = predArray_to_predMatrix(pred_array, config.target_size)

	# Conver to binary images. Apply morphological transformations.
	# Convert to binary image just in case using a semantic segmetnation model with many classes
	truth_reshaped = apply_morph(truth_reshaped, morph_type=None) # Just converts to binary image
	pred_reshaped = apply_morph(pred_reshaped, morph_type='foreground')

	determine_segmentation_accuracy(truth_reshaped, pred_reshaped, config, radius, base_output_path)


def determine_segmentation_accuracy(truth_matrix, pred_matrix, config, radius, base_output_path):
	"""
	Returns the accuracy in identifying the location of a particle, when comparing truth_matrix with pred_matrix
	Does not look at a each pixel. Instead, determines if a particle has been more broadly identified by the model. 
	Arguments 
	truth_matrix: Binary matrix with segmented pixels (0 is background, and 1 is foreground)
	pred_matrix: Predicted binary matrix with segmented pixels (0 is background, and 1 is foreground)
	Guiding heuristics
	+ For each ground truth particle, the algorthm determines if there is a predicted particle in the region of interest. The same predicted particle blob can  be used twice for different ground truth particles. 
	+ Possible Improvement: A current issue is that ground truth and predicted particles merge into a single particle. The connectedComponentsWithStats treats these merged particles as a single particle, leading to errors both in the wrongly detected and missed detections. A solution is to look at any pixel in region of interest, instead of just looking at the centroid. 
	+ Possible Improvement: Instead of using connectedComponentsWithStats for the ground truth, use the ground truth coordinates to seed the location of the ground truth particles. 
	"""

	# Transform colors for visualization (converts from single to three channels)
	truth_output = get_color_image(truth_matrix, nclasses = 2, colors = [(0, 0, 0), (128,128,128)])
	pred_output = get_color_image(pred_matrix, nclasses = 2, colors = [(0, 0, 0), (128,128,128)])

	# Input requirement: 8-bit single, channel image. Image input is binary with all non-zero pixels treated as 1s. 
	# connected_output array: [num_labels, label_matrices, marker_stats, centroids]
	# Connectivity: Connectivity of 8 makes it more likely that particles that are merged due to proximity will be treated seperately. 
	truth_connected = cv2.connectedComponentsWithStats(truth_matrix.astype('int8'), connectivity=8)
	pred_connected = cv2.connectedComponentsWithStats(pred_matrix.astype('int8'), connectivity=8)
	truth_centroids = np.asarray(truth_connected[3][1:])  # Remove the background centroid (at index 0).
	pred_centroids = np.asarray(pred_connected[3][1:]) # Remove the background centroid (at index 0).
	truth_intersection_indices = [] # List of indecies that are in both ground truth and predictoin set
	pred_intersection_indices = [] # List of indecies that are in both ground truth and predictoin set

	# Measured stats 
	total_ground_truth_particles = truth_centroids.shape[0]
	total_pred_particles = pred_centroids.shape[0]
	intersection_of_pred_and_truth = 0
	only_truth_particles = 0
	only_pred_particles = 0

	# Determine the intersection between the predicted particles and the ground truth particles. (GREEN)
	for index_truth, target_centroid_truth in enumerate(truth_centroids): 


		nearest_i_pred_list, nearest_distance_list = centroids_within_radius(target_centroid_truth, pred_centroids, radius)

		# Determine the ground truth particles successfully detected in the predicted set. 
		# The intersection of ground truth and prediction sets. 
		for i, nearest_i_pred in enumerate(nearest_i_pred_list):
			if nearest_distance_list[i] < radius: # Sanity check centroids_within_radius (needed for nearest_centroid)
				intersection_of_pred_and_truth += 1

				truth_intersection_indices.append(index_truth)
				pred_intersection_indices.append(nearest_i_pred)

				# Debug: Place GREEN indicator for each ground truth partidcle detected. 
				if (config.debug):
					centroid_circle = (int(target_centroid_truth[0]), int(target_centroid_truth[1]))
					cv2.circle(truth_output, center=centroid_circle, radius=radius, color=(0, 255, 0), thickness=3)
					cv2.circle(pred_output, center=centroid_circle, radius=radius, color=(0, 255, 0), thickness=3)


	# Determine particles only in ground truth set (RED)
	# Repeat indices are just removed once (do not cause issues)
	truth_centroids = np.delete(truth_centroids, truth_intersection_indices, axis=0) 
	for target_centroid_truth in truth_centroids:
		only_truth_particles += 1

		# Debug: Place RED indicator for particles only in ground truth set. 
		if (config.debug):
			centroid_circle = (int(target_centroid_truth[0]), int(target_centroid_truth[1]))
			cv2.circle(truth_output, center=centroid_circle, radius=radius, color=(0, 0, 255), thickness=3)
			cv2.circle(pred_output, center=centroid_circle, radius=radius, color=(0, 0, 255), thickness=3)


	# Determine particles only in prediction set. (BLUE)
	pred_centroids = np.delete(pred_centroids, pred_intersection_indices, axis=0)
	for target_centroid_pred in pred_centroids: 
		only_pred_particles += 1

		# Debug: Place BLUE indicator for particlesonly in prediction set.
		if (config.debug):
			centroid_circle = (int(target_centroid_pred[0]), int(target_centroid_pred[1]))
			cv2.circle(truth_output, center=centroid_circle, radius=radius, color=(255, 0, 0), thickness=3)
			cv2.circle(pred_output, center=centroid_circle, radius=radius, color=(255, 0, 0), thickness=3)

		
	# Print results
	config.logger.info("\nParticle Detection Accuracy for: %s", base_output_path)
	config.logger.info("Total particles in ground truth set: %d", total_ground_truth_particles)
	config.logger.info("Total particles in prediction set: %d", total_pred_particles)
	config.logger.info("Total particles in prediction set that map to truth set: %d", intersection_of_pred_and_truth)
	config.logger.info("Only ground truth set: %d", only_truth_particles)
	config.logger.info("Only prediction set: %d", only_pred_particles)
	config.logger.info("\n")


	# Debug: Save images with indicators. 
	if (config.debug):
		cv2.imwrite(base_output_path + "_truth.jpg", truth_output)
		cv2.imwrite(base_output_path + "_pred.jpg", pred_output)

def nearest_centroid(centroid, array_of_centroids):
	"""
	Calculates the closest point in array_of_centroids to centroid. 
	Args
	array_of_centroids: Needs to be a numpy array, with same format as centroid. 
	centroid: Tuple with x and y coordinates. 
	"""

	# No nearest centroids since none predicted by model. 
	if (len(array_of_centroids) == 0): 
		nearest_i = -1 
		nearest_distance = float('inf') # guarantees that particle marked as not identified. 
	else: 
		dist_2 = np.sum((array_of_centroids - centroid)**2, axis=1) # For comparison purposes, don't need to calculate actual distnace. 
		nearest_i =  np.argmin(dist_2)
		nearest_distance = math.sqrt(dist_2[nearest_i])

	return nearest_i, nearest_distance

def centroids_within_radius(centroid, array_of_centroids, radius):
	"""
	Calculates the points i array_of_centroids that are within radius of centroid. 
	Args
	array_of_centroids: Needs to be a numpy array, with same format as centroid. 
	centroid: Tuple with x and y coordinates. 
	"""


	nearest_i_list = []
	nearest_distance_list = []
	dist_2 = np.sum((array_of_centroids - centroid)**2, axis=1) # For comparison purposes, don't need to calculate actual distnace. 

	within_radius_mask = (dist_2<= radius**2)

	nearest_i_list = within_radius_mask.nonzero()[0] # nonzero returns a tuple with indices for each dimensino. Extract only dimension
	nearest_distance_list = np.sqrt(dist_2[nearest_i_list])


	return list(nearest_i_list), list(nearest_distance_list)

def get_crop_coordinates(input_image_shape, centroid, target_dim):
	"""
	Crop input_image around the centroid based on the target_dim. 
	The output crop should have target_dim*target_dim dimensions, or smaller if constrained by input_image dimensions. 
	"""

	max_height =input_image_shape[0] # y-dimensions
	max_width = input_image_shape[1] # x-dimension

	# Obtain cropping dimensions
	x1 = int(centroid[0]-target_dim/2)
	y1 = int(centroid[1]-target_dim/2)
	x2 = int(centroid[0]+target_dim/2)
	y2 = int(centroid[1]+target_dim/2)

	# Ensure cropping dimensions within dimensions of input_image
	x1 = x1 if x1>0 else 0
	x2 = x2 if x2<max_width else max_width
	y1 = y1 if y1>0 else 0
	y2 = y2 if y2<max_height else max_height


	return x1,x2,y1,y2


def load_model(model, path, config):

	if path is  None: 
		config.logger.info("Attempted loading custom weights. Load file set to None.")
		return 


	model.load_weights(path)
	config.logger.info("Load Model from %s", path)



def save_model(model, path, config):
	if path is  None: 
		config.logger.info("Attempted save model weights. No save location given in configuration.")
		return 

	model.save(path)
	config.logger.info("Save Model to %s", path)

def save_categorical_aray_as_image(img_array, path, config):
	img_labeled = np.argmax(img_array, axis=1) # Convert from categorical format to label format. 
	img_input = np.reshape(img_labeled, config.target_size)

	img_output = get_color_image(img_input, config.nclasses, config.colors)

	cv2.imwrite(path, img_output ) # use cv2 since data is BGR and a numpy array



def get_color_image(img_input, nclasses, colors):

	# Build outpush data structure
	img_output = np.zeros(img_input.shape + (3,))

	# Assign colors to each class
	# Note: Need to add results into image so don't replace previous results. 
	for c in range(nclasses):
		img_output[:,:,0] += (img_input == c)*colors[c][0]
		img_output[:,:,1] += (img_input == c)*colors[c][1]
		img_output[:,:,2] += (img_input == c)*colors[c][2]

	return img_output


def delete_folder_with_confirmation(folder_path):
	"""
	Description: If the folder exists, delete if recursively after user confirmation
	"""
	if (os.path.exists(folder_path)): 
		user_question = 'Delete %s: (yes/no): '%(folder_path)
		reply = str(raw_input(user_question)).lower().strip()
		if (reply == 'yes'):
			shutil.rmtree(folder_path)
		else:
			raise RuntimeError("Output folder already exists. However, user indicate not to delete output folder.")



def print_summary_statistics_for_labels(label_list, class_mapping, config, discard_label= None, image_count = 1):
	"""
	Description: Prints summary of results
	Args
	class_mapping: dict that maps {label (int): class name (str)}
	config: Instance of either segmentation or classificatino configuration object. 
	discard_lavel: The label (int) to discard within summary. Usually 'other' or 'background' class
	image_count: The total number of canvas images used to get the labels in label_list
	"""

	# Get counts for each of the particle labels
	label_count_dic = count_labels(label_list, class_mapping)
	# Count total number of labels
	total_labels = sum(label_count_dic.values())
	# Count the total number of particles => discarding the discard_label
	total_particles = np.sum(counts for label, counts in label_count_dic.iteritems() if label != discard_label)

	# Print summary results
	config.logger.info("Results")
	config.logger.info("Total Labels: %d", total_labels)
	config.logger.info("Total Particles (discared label %d): %d", discard_label, total_particles)
	config.logger.info("Type\t\t\tCount\t\t\tperPrimas\t\t\tperHPF\t\t\tPercent of Particles")


	# Print out results for each class
	k_primas_to_HPF = config.hpf_area/float(config.primas_area) # converts from Primas unit to HPF unit. 
	for label, class_name in class_mapping.iteritems():
		# Calculate metrics
		perPrimas = label_count_dic[label]/float(image_count) #calculates average per primas microscope image
		perHPF = perPrimas*k_primas_to_HPF # converts to perHPF
		particle_percent = 100*label_count_dic[label]/float(total_particles)

		# Print results
		# Pretty print the percent
		pp_percent = '%.02f%%'%(particle_percent)
		if label == discard_label:
			pp_percent = "N/A"
		# Print single row of results
		print_row_str = "%s\t\t\t%d\t\t\t%.02f\t\t\t\t%.02f\t\t\t%s"
		config.logger.info(print_row_str, class_name, label_count_dic[label], perPrimas, perHPF, pp_percent)


def count_labels(label_list, class_mapping):
	"""
	Description: Counts the occurrence of labels in label_list. Outputs the result as a dict. 
	Returns
	label_count_dic: A dictionary that maps the labels to the number of counts of that label in label_list
	"""
	# Initialize the dictionary
	label_count_dic = {}
	for label in class_mapping:
		label_count_dic[label] = 0

	for label in label_list: 
		if label in label_count_dic:
			label_count_dic[label] += 1
		else:
			raise ValueError("count_labels: Label in label_list doesn't map with labels given in class_mapping.")
			

	return label_count_dic


def get_file_name_from_path(target_file_path, remove_ext=True):
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


def get_coordinates_from_cropname(crop_filename):
	"""
	crop_filename: Extract the coordinate positions from crop_filename. 
	Args
	crop_filename: The filename of the cropped image that contains coordinates embedded in the name. 
	+ Currently, the standard format is 'img1_43_235.bmp' or 'img1_43_235_cpy20.bmp' or '10um_particle_40_1455_cpy304.bmp'
	image_dim: A tuple that indicates the dimensions of an image. The tuple is given in (height, width) format.
		Only calculate the delta_pos and angular_pos with image_dim given 
	Return 
	coordinate_pos: The raw coordinate position with the origin at pixel (0,0). (given in (widht, height format))
	"""

	# Identify particle centroid from crop image name
	coordinates_str = re.findall("_(\d+)", crop_filename)
	# Provided in (widht, height) format
	coordinate_pos = [float(coordinates_str[0]), float(coordinates_str[1])]


	return coordinate_pos

def initialize_classification_model(log_dir=None):
	""" 
	Builds the classification  model and loads the pre-trained weights. 
	"""

	# Instantiates configuration for training/validation
	config = ClassifyParticles_config.ClassifyParticles_Config()

	# Reroute/update logging
	if (log_dir):
		prefix = log_dir.split('/')[-2]
		log_file_name = prefix + "_classification_prediction.log" #If None, then name based on datetime.
		config.logger = create_logger(log_dir, file_name=log_file_name, log_name="predict_classify_logger")


	# Print configuration
	print_configurations(config) # Print config summary to log file


	# Instantiate training/validation data
	data = ClassifyParticlesData.ClassifyParticlesData(config)

	# Builds model 
	model = createModel.base_model_with_pos(input_shape = config.image_shape, base_weights = config.imagenet_weights_file, classes=config.nclasses)


	# Load weights (if the load file exists)
	load_model(model, config.weight_file_input, config)

	return model, data

