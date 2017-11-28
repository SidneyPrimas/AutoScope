"""
Referenced from: https://github.com/DeepLearningSandbox/DeepLearningSandbox/tree/master/transfer_learning
"""

# Import basic libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import urllib
from PIL import Image
import glob
import os
import math

#Import keras libraries
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import SGD

# Import Model
from model_factory import VGG16_with_custom_FC1

# Configuration variables
TARGET_SIZE = (224, 224)
CHANNELS = 1
IMAGE_SHAPE = TARGET_SIZE + (CHANNELS,)
BATCH_SIZE = 32
EPOCH = 1
TRAIN_DIR = "./data/random_datasets/cat_dog_data/train_dir"
VAL_DIR =  "./data/random_datasets/cat_dog_data/val_dir"
CLASSES = len(glob.glob(TRAIN_DIR + "/*"))
INPUT_WEIGHTS_FILE = "./model_storage/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
OUTPUT_MODEL_FILE = "./model_storage/v2_classification/random_datasets/cat_dog_data/20171125_vgg16_customTF_dev.h5"
TL_FREEZE_LAYER = "block5_pool"


def get_img_count(directory):
	"""
	Walk through sub-directories of directory path. Count all images in sub-folders. 
	Args: 
	directory: Path to directory folder containing classes. Each sub-folder is a class and only contains images. 
	Returns:
	Count of total images in directory (across all classes)
	"""
	if not os.path.exists(directory):
		return 0
	cnt = 0
	# Identify each sub-directory in directory (classes)
	for r, dirs, _ in os.walk(directory):
		# Count images in sub-directories. 
		for dr in dirs:
			cnt += len(glob.glob(os.path.join(r, dr + "/*")))
	return cnt

def get_batches_per_epoch(directory, batch_size):
	"""
	Args: 
	directory: Root directory for folder (includes sub-directories with all the classes)
	batch_size: The number of files per batch. 
	Returns: 
	Number of epochs needed to iterate through an entire dataset with a given batch_size
	"""
	num_files = get_img_count(directory)
	# Need to cast so don't truncate results due to int division. 
	# Use ceil to make sure a single epoch represents a full-cylce of all the images. 
	return math.ceil(num_files/float(batch_size)) 

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
		layer.trainable = train_this_layer

		# Train layers above layer_name
		if layer.name == layer_name: 
			train_this_layer = True

def plot_training(history):
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(len(acc))

	plt.plot(epochs, acc, 'r.')
	plt.plot(epochs, val_acc, 'r')
	plt.title('Training and validation accuracy')

	plt.figure()
	plt.plot(epochs, loss, 'r.')
	plt.plot(epochs, val_loss, 'r-')
	plt.title('Training and validation loss')
	plt.show()


def train():
	"""

	"""
	batches_per_epoch_train =  get_batches_per_epoch(TRAIN_DIR, BATCH_SIZE)
	print "batches_per_epoch_train: %d"%(batches_per_epoch_train)
	batches_per_epoch_val = get_batches_per_epoch(VAL_DIR, BATCH_SIZE)
	print "batches_per_epoch_val: %d"%(batches_per_epoch_val)

	# Build model
	model = VGG16_with_custom_FC1(input_shape = IMAGE_SHAPE, base_weights = INPUT_WEIGHTS_FILE, classes=CLASSES)

	# Freeze Lower Layers
	freeze_lower_layers(model, TL_FREEZE_LAYER)


	# Compile Model
	# Note on loss: Loss value minimized by individual model is the sum of each model. 
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	model.summary()

	# Transformation randomly applied to each image. DataGenerator loops over all images indefinitely, and applies transformation IRL. 
	# Use fit() with ImageDataGenerator in order to compute data-dependent transformations, based on an array of sampled data. Required for featurewise_center or featurewise_std_normalization or zca_whitening.
	train_datagen =  ImageDataGenerator(
		preprocessing_function=preprocess_input,
		rotation_range=30,  # Preprocess function applied to each image before any other transformation. Applies normalization.
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True
	)

	# For the test dataset, do not apply data augmentation. Only use the original data. 
	test_datagen = ImageDataGenerator(
		preprocessing_function=preprocess_input, # Preprocess function applied to each image before any other transformation. Applies normalization. 
	)

	# Takes path to directory and outputs batches of augmented normailized data. Does this indefinitely. 
	# Use train_generator.class_indices to get dictionary that maps classes to indicies. 
	train_generator = train_datagen.flow_from_directory(
		TRAIN_DIR,
		target_size= TARGET_SIZE,
		batch_size=BATCH_SIZE,
		shuffle = True, # already default
		color_mode = "rgb", #already default
		#save_to_dir = "" # Used to visualize data augmentation of images. Use save_prefix to indicate prefix of images. 
		#classes = ['dogs', 'cats'] # Currently, classes are inferred from sub-directories. 
	)


	validation_generator = test_datagen.flow_from_directory(
		VAL_DIR,
		target_size=TARGET_SIZE, # Auto-resize of all images. 
		batch_size=BATCH_SIZE,
		shuffle = True,# default already true. 
		color_mode = "rgb", #already default
	)



	# Fits model to data provided by generator. 
	# Data generator (including augmentation) can run in parallel to fitting. 
	# Data generator produced batch-by-batch dataset. 
	# Can use an array of images, batch and generator for training, validation and prediction. 
	history_tl = model.fit_generator(
		train_generator, # Calling the function once produced a tuple with (inputs, targets, weights). Produces training data for a single batch. 
		epochs=EPOCH,
		steps_per_epoch= batches_per_epoch_train, # (e.g.: batches_per_epoch) The total batches processes to complete an epoch. steps_per_epoch = total_images/batch_size
		validation_data=validation_generator, # Produces validation data. 
		validation_steps= batches_per_epoch_val
	)

	model.save(OUTPUT_MODEL_FILE)

	print history_tl.history
	plot_training(history_tl)


if __name__=="__main__":
	print "Image Size: %s"%(str(TARGET_SIZE))
	print "Image Formatting: %s"%(K.image_data_format())
	print "Training Files; %d"%(get_img_count(TRAIN_DIR))
	print "Validation Files; %d"%(get_img_count(VAL_DIR))
	print "Class Size: %d"%(CLASSES)

	if (not os.path.exists(TRAIN_DIR)) or (not os.path.exists(VAL_DIR)):
		print("directories do not exist")
		sys.exit(1)

	train()

