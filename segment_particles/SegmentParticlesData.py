# Import basic libraries
import numpy as np
import matplotlib.pyplot as plt
import PIL
import glob
import os
import math
from datetime import datetime
from collections import defaultdict
import random
import itertools
import cv2


# Import keras libraries
from tensorflow.python.keras.preprocessing import image as image_keras

# Import local libraries
import CNN_functions
from SegmentParticles_config import SegmentParticles_Base


class SegmentParticlesData(object):

	# Create instance variable since config passed to IrisData instance. 
	def __init__(self, config):
		self.config = config
		self.train_count = 0 
		self.image_train_count = 0 

		# Create data generators
		self.train_generator = self._create_training_generator()
		self.val_generator = self._create_validation_generator()


	def _get_batches_per_epoch(self, directory, batch_size):
		"""
		Args: 
		directory: Root directory for folder (includes sub-directories with all the classes)
		batch_size: The number of files per batch. 
		Returns: 
		Number of epochs needed to iterate through an entire dataset with a given batch_size
		"""
		num_files = self.get_img_count(directory)
		# Need to cast so don't truncate results due to int division. 
		# Use ceil to make sure a single epoch represents a full-cylce of all the images. 
		return math.ceil(num_files/float(batch_size)) 



	def _make_data_generator(self, image_dir_path, annotations_dir_path):

		# Symmetrically list of images and annotations 
		images_list = glob.glob(image_dir_path + "*.bmp")
		annotations_list = glob.glob(annotations_dir_path + "*.bmp")
		images_list.sort()
		annotations_list.sort()
		reorder =  range(len(images_list))
		random.shuffle(reorder)
		images_list = [images_list[i] for i in reorder]
		annotations_list = [annotations_list[i] for i in reorder]

		# Verify Assumptions: image_list and annotations_list have the same order with the same file names. 
		assert(len(images_list) == len(annotations_list))
		for i in range(len(images_list)):
			image_name = images_list[i].split('/')[-1]
			annotation_name = annotations_list[i].split('/')[-1]
			assert(image_name == annotation_name)

		#  Returns generator that cycles through the list of images/annotations
		images_iterator = itertools.cycle(images_list)
		annotations_iterator = itertools.cycle(annotations_list)

		# Provide an image and the corresponding annotations. 
		while(True): 
			x_input = []
			y_target = []

			for i in range(self.config.batch_size):

				image_path = next(images_iterator)
				annotation_path = next(annotations_iterator)

				# Get images
				x_input.append(self._get_image(image_path))
				y_target.append(self._get_annotation(annotation_path))

			yield np.array(x_input), np.array(y_target)


	def _get_image(self, image_path):
		"""
		Loads and resizes image. Zero centers each pixel. Converts to 32-float BGR dataset. 
		"""
		img = PIL.Image.open(image_path) # Use PIL since in correct RGB format. And, Keras relies on PIL. 

		# Assumed format: (height, width, dimension) 
		if img.size != self.config.target_size:
			img = img.resize(self.config.target_size, resample = PIL.Image.ANTIALIAS) #PIL implementation: Use antialias to not alias while downsampling. 
		
		x = image_keras.img_to_array(img) # convert to numpy array (as float 32)

		# Zero center images based on imagenet 
		# 'RGB'->'BGR' (PIL provided RGB input)
		x = x[..., ::-1]
		# Zero-center by mean pixel (based on VGG16 means)
		x[..., 0] -= 103.939
		x[..., 1] -= 116.779
		x[..., 2] -= 123.68

		return x




	def _get_annotation(self, annotation_path):
		"""
		Loads and resizes the annotation. Transforms the annotation into results tensor compatible with a softmax output. 
		"""
		img = PIL.Image.open(annotation_path)  # Use PIL since in correct RGB format. And, Keras relies on PIL.

		# Assumed format: (height, width, dimension) 
		if img.size != self.config.target_size:
			#PIL implementation: Bilinear used since 1) creates more fore-ground pixels (better for training) and 2) full-cirlces (instead of spotted circles).
			img = img.resize(self.config.target_size, resample = PIL.Image.BILINEAR) 

		x = image_keras.img_to_array(img) # convert to numpy array (as float 32)
		x = x[..., ::-1] # 'RGB'->'BGR' (PIL provided RGB input)

		# Convert to a softmax labeling structure. 
		# Each pixel needs to be labeld with a single class. 
		# To label a pixel with 5 classes: [0,0,0,1,0] means pixels is label 4. 
		# Implementation note: Only works with 2 classes => forground/background. 
		img_labels = np.zeros(self.config.target_size + (self.config.nclasses,))
		for m in range(x.shape[0]):
			for n in range(x.shape[1]):
				pixel_val = x[m,n,0] # Take value from blue channel 
				if pixel_val == 0:
					img_labels[m,n,0] = 1 # Background label
				else:
					img_labels[m,n,1] = 1 # Foreground label

		# Reshape to model output. 
		img_labels = np.reshape(img_labels, ( self.config.target_size[0]*self.config.target_size[1] , self.config.nclasses ))
		return img_labels


	def _create_training_generator(self): 

		#  If not already, initialize batches per epoch. 
		if (self.config.batches_per_epoch_train == None):
			self.config.batches_per_epoch_train = self._get_batches_per_epoch(self.config.train_images_dir, self.config.batch_size)
		self.config.logger.info("Batches per Epoch for Training: %d",(self.config.batches_per_epoch_train))

		return self._make_data_generator(self.config.train_images_dir, self.config.train_annotations_dir)


	def _create_validation_generator(self):

		# If not already, initialize batches per epoch. 
		if (self.config.batches_per_epoch_val == None):
			self.config.batches_per_epoch_val = self._get_batches_per_epoch(self.config.val_images_dir, self.config.batch_size)
		self.config.logger.info("Batches per Epoch for Validation: %d",(self.config.batches_per_epoch_val))

		return self._make_data_generator(self.config.val_images_dir, self.config.val_annotations_dir)

	def _save_validation_images(self, original_img, truth_array, pred_array, img_num):
		# orgingal
		path = self.config.output_img_dir + str(self.image_train_count) + "_" + str(img_num) + "_original.jpg"
		cv2.imwrite(path, original_img )
		# truth
		path = self.config.output_img_dir + str(self.image_train_count) + "_" + str(img_num) + "_truth.jpg"
		CNN_functions.save_categorical_aray_as_image(truth_array, path, self.config)
		# prediction 
		path = self.config.output_img_dir + str(self.image_train_count) + "_" + str(img_num) + "_prediction.jpg"
		CNN_functions.save_categorical_aray_as_image(pred_array, path, self.config)


	def print_data_summary(self): 
		self.config.logger.info("###### DATA ######")
		self.config.logger.info("Total Training Data: %d", len(glob.glob(self.config.train_images_dir + '*.bmp')))
		self.config.logger.info("Total Validation Data: %d", len(glob.glob(self.config.val_images_dir + '*.bmp')))
		self.config.logger.info("###### DATA ######  \n\n")

	def validate_epoch(self, model): 
		""" 
		Validates model for a single epoch. Provides average results across the entire epoch. 
		"""

		# Track overall label metrics
		all_truth = None
		all_pred = None

		# Validate across multiple batches
		for _ in range(self.config.batches_per_epoch_val):
			img_input, label_truth =  next(self.val_generator)
			label_pred = model.predict_on_batch(img_input)
			
			# Append to overall metrics
			if all_truth is None: 
				all_truth = label_truth
				all_pred = label_pred
			else: 
				all_truth = np.append(all_truth, label_truth, axis=0)
				all_pred = np.append(all_pred, label_pred, axis=0)


		accuracy =CNN_functions.get_accuracy(all_truth, all_pred)

		# Save random image from last batch (original, truth, prediction). Once per epoch
		if (self.config.debug): 
			for _ in range(1): 
				img_num =  random.randint(0, self.config.batch_size-1)
				self._save_validation_images(
					original_img= img_input[img_num,:,:],
					truth_array= label_truth[img_num,:], 
					pred_array= label_pred[img_num,:], 
					img_num= img_num)

		# Output results
		self.config.logger.info("Validation Results")
		self.config.logger.info("Validation accuracy: %1.3f" %(accuracy))
		self.config.logger.info("\n\n")

	def train_epoch(self, model, in_house = True):
		""" 
		Trains model for a single epoch. Provides average results across the entire epoch. 
		"""

		# Track overall metrics
		loss = 0
		accuracy = 0

		# Use home-brew training
		if (in_house): 
			loss_all = []
			accuracy_all = []
			for batch in range(self.config.batches_per_epoch_train):
				img_input, label_truth =  next(self.train_generator)
				metric_output = model.train_on_batch(img_input, label_truth)
				# Update per batch tracking variables
				self.train_count += 1
				self.image_train_count += self.config.batch_size
				loss_all.append(metric_output[0])
				accuracy_all.append(metric_output[1])

			loss = np.mean(loss_all)
			accuracy = np.mean(accuracy_all)


		# Use keras-built in training
		else: 
			history = model.fit_generator(
				self.train_generator, # Produces training data for a single batch. 
				epochs = 1, # Only train a single epoch 
				steps_per_epoch=  self.config.batches_per_epoch_train # The total batches processes to complete an epoch. steps_per_epoch = total_images/batch_size
			)
			self.train_count += self.config.batches_per_epoch_train # Update image tracking 
			self.image_train_count += self.config.batches_per_epoch_train*self.config.batch_size

			# Note: Use mean just in case we increase epochs in model.fit_generator
			loss = np.mean(history.history["loss"])
			accuracy = np.mean(history.history["acc"])

		# Output results
		self.config.logger.info("Training Results")
		self.config.logger.info("Step: %d, Images Trained: %d, Batch loss: %2.6f, Training accuracy: %1.3f" %(self.train_count, self.image_train_count, loss, accuracy))


	def train_entire_model(self, model): 
		# Compiles model
		# Note: categorical_crossentropy requires prediction/truth data to be in categorical format. 
		model.compile(optimizer=self.config.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		model.summary(print_fn = self.config.logger.info)

		while True:
			self.config.logger.info("######   Entire Model Training  ######")
			self.train(model)
			CNN_functions.save_model(model, self.config.weight_file_output + str(self.image_train_count) + ".h5", self.config) # Save


	def train(self, model): 

		for epoch in range(self.config.num_epochs): 

			# Training for single epoch
			self.train_epoch(model, in_house = False)

			# Validation for single epoch
			self.validate_epoch(model)




