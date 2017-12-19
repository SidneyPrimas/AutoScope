# Import basic libraries
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import json
import os
import cv2
import math

from tensorflow.python.keras._impl.keras import backend as K


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
		 

def get_data_config(log_path):
	log = open(log_path, 'r')
	data_config = json.load(log)
	log.close()

	return data_config

def validate_config(config):
	assert(config.nclasses == config.data_config["nclasses"]) # Check that the data creation nclasses is the same as the model nclasses. 



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


def get_foreground_accuracy_perImage(truth_array, pred_array, config, radius, img_num):
	"""
	Returns the systems accuracy in identifying the location of a particle.
	Does not look at a each pixel. Instead, determines if a particle has been more broadly identified by the model. 
	Arguments 
	truth_array: Single ground truth array in (image_pixels, classes)
	pred_array: Single predictions array in (image_pixels, classes)
	Guiding heuristics
	+ For each ground truth particle, the algorthm determines if there is a predicted particle in the region of interest. The same predicted particle blob can  be used twice for different ground truth particles. 
	+ Possible Improvement: A current issue is that ground truth and predicted particles merge into a single particle. The connectedComponentsWithStats treats these merged particles as a single particle, leading to errors both in the wrongly detected and missed detections. A solution is to look at any pixel in region of interest, instead of just looking at the centroid. 
	+ Possible Improvement: Instead of using connectedComponentsWithStats for the ground truth, use the ground truth coordinates to seed the location of the ground truth particles. 
	"""
	# Convert from categorical format to label format. 
	truth_labeled = np.argmax(truth_array, axis=1) 
	pred_labeled = np.argmax(pred_array, axis=1) 
	# Reshape into single channel images. 
	truth_reshaped = np.reshape(truth_labeled, config.target_size)
	pred_reshaped = np.reshape(pred_labeled, config.target_size)


	# Convert images to binary (if there are multiple classes)
	truth_reshaped = ((truth_reshaped > 0)).astype('uint8')
	pred_reshaped = ((pred_reshaped > 0)).astype('uint8')

	
	# Erosions
	# Note: Erosions allows for 1) seperation of merged blobs and 2) removal of popcorn prediction noise. 
	struct_element = np.ones((2,2), np.uint8)
	pred_reshaped = cv2.erode(pred_reshaped.astype('float32'), struct_element, iterations=1)

	# Transform colors for visualization: color as binary images
	truth_output = get_color_image(truth_reshaped, nclasses = 2, colors = [(0,0,0), (128,128,128)])
	pred_output = get_color_image(pred_reshaped, nclasses = 2, colors = [(0,0,0), (128,128,128)])

	# Input requirement: 8-bit single, channel image. Image input is binary with all non-zero pixels treated as 1s. 
	# connected_output array: [num_labels, label_matrices, marker_stats, centroids]
	# Connectivity: Connectivity of 8 makes it more likely that particles that are merged due to proximity will be treated seperately. 
	truth_connected = cv2.connectedComponentsWithStats(truth_reshaped.astype('int8'), connectivity=8)
	pred_connected = cv2.connectedComponentsWithStats(pred_reshaped.astype('int8'), connectivity=8)
	truth_centroids = np.asarray(truth_connected[3][1:])  # Remove the background centroid (at index 0).
	pred_centroids = np.asarray(pred_connected[3][1:]) # Remove the background centroid (at index 0).
	truth_intersection_indices = [] # List of indecies that are in both ground truth and predictoin set
	pred_intersection_indices = [] # List of indecies that are in both ground truth and predictoin set

	# Measured stats 
	total_ground_truth_particles = truth_centroids.shape[0]
	intersection_of_pred_and_truth = 0
	only_truth_particles = 0
	only_pred_particles = 0

	# Determine the intersection between the predicted particles and the ground truth particles. (GREEN)
	for index_truth, target_centroid_truth in enumerate(truth_centroids): 

		# Calculate the nearest centroid
		nearest_i_pred, nearest_distance = nearest_centroid(target_centroid_truth, pred_centroids)

		# Determine the ground truth particles successfully detected in the predicted set. 
		# The intersection of ground truth and prediction sets. 
		if nearest_distance < radius:
			intersection_of_pred_and_truth += 1

			truth_intersection_indices.append(index_truth)
			pred_intersection_indices.append(nearest_i_pred)

			# Debug: Place GREEN indicator for each ground truth partidcle detected. 
			if (config.debug):
				centroid_circle = (int(target_centroid_truth[0]), int(target_centroid_truth[1]))
				cv2.circle(truth_output, center=centroid_circle, radius=radius, color=(0, 255, 0), thickness=1)
				cv2.circle(pred_output, center=centroid_circle, radius=radius, color=(0, 255, 0), thickness=1)


	# Determine particles only in ground truth set (RED)
	# Repeat indices are just removed once (do not cause issues)
	truth_centroids = np.delete(truth_centroids, truth_intersection_indices, axis=0) 
	for target_centroid_truth in truth_centroids:
		only_truth_particles += 1

		# Debug: Place RED indicator for particles only in ground truth set. 
		if (config.debug):
			centroid_circle = (int(target_centroid_truth[0]), int(target_centroid_truth[1]))
			cv2.circle(truth_output, center=centroid_circle, radius=radius, color=(0, 0, 255), thickness=1)
			cv2.circle(pred_output, center=centroid_circle, radius=radius, color=(0, 0, 255), thickness=1)


	# Determine particles only in prediction set. (BLUE)
	pred_centroids = np.delete(pred_centroids, pred_intersection_indices, axis=0)
	for target_centroid_pred in pred_centroids: 
		only_pred_particles += 1

		# Debug: Place BLUE indicator for particlesonly in prediction set.
		if (config.debug):
			centroid_circle = (int(target_centroid_pred[0]), int(target_centroid_pred[1]))
			cv2.circle(truth_output, center=centroid_circle, radius=radius, color=(255, 0, 0), thickness=1)
			cv2.circle(pred_output, center=centroid_circle, radius=radius, color=(255, 0, 0), thickness=1)

		
	# Print results
	config.logger.info("\nParticle Detection Accuracy for: Img_num %d", img_num)
	config.logger.info("Total particles in ground truth set: %d", total_ground_truth_particles)
	config.logger.info("Percent of ground truth particles detected: %f", intersection_of_pred_and_truth/float(total_ground_truth_particles))
	config.logger.info("Intersection between prediction and ground truth sets: %d", intersection_of_pred_and_truth)
	config.logger.info("Only ground truth set: %d", only_truth_particles)
	config.logger.info("Only prediction set: %d", only_pred_particles)
	config.logger.info("\n")


	# Debug: Save images with indicators. 
	if (config.debug):
		cv2.imwrite(config.output_img_dir + str(img_num) + "_truth.jpg", truth_output)
		cv2.imwrite(config.output_img_dir + str(img_num) + "_pred.jpg", pred_output)

def nearest_centroid(centroid, array_of_centroids):
	"""
	Calculates the closest point in array_of_centroids to centroid. 
	Args
	array_of_centroids: Needs to be a numpy array, with same format as centroid. 
	centroid: Tuple with x and y coordinates. 
	"""
	dist_2 = np.sum((array_of_centroids - centroid)**2, axis=1) # For comparison purposes, don't need to calculate actual distnace. 
	nearest_i =  np.argmin(dist_2)
	nearest_distance = math.sqrt(dist_2[nearest_i])

	return nearest_i, nearest_distance

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

		





