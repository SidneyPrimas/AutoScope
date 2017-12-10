# Import basic libraries
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import json
import os
import cv2

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

def get_accuracy(all_truth, all_pred):
	'Return the accuracy given the predicted labels and the ground truth labels. '
	correct_prediction = np.equal(np.argmax(all_truth, axis=2), np.argmax(all_pred, axis=2))
	accuracy = np.mean(correct_prediction.astype(np.float))

	return accuracy

def load_model(model, path, config):

	if path is  None: 
		config.logger.info("Attempted loading custom weights. Load file set to None.")
		return 


	model.load_weights(path)
	config.logger.info("Load Model from %s", path)



def save_model(model, path, config):
	if path is  None: 
		return 

	model.save(path)
	config.logger.info("Save Model to %s", path)

def save_categorical_aray_as_image(img_array, path, config):
	img_labeled = np.argmax(img_array, axis=1) # Convert from categorical format to label format. 
	img = np.reshape(img_labeled, config.target_size)
	img = img*255 # Convert from class_label to greyscale color. 
	cv2.imwrite(path, img ) # use cv2 since data is BGR and a numpy array


# Possible future function. 
"""
def crop( o1 , o2 , i  ):
Description: Crops o1 (the upsampled image) or o2 (the featue map carried over from the encoding layers) depending on the larger image. This implementation allows for different size input images. 
Problem: The image is not cropped from the center. Instead, it's cropped starting from the top left corner. 
Args: 
o1: Image to crop (usually the upsampled image)
o2: Image used as a reference (usually the feature map from the encoding layers. )
i: Input layer
Returns: 
The cropped o1 and o2 images. 

"""




