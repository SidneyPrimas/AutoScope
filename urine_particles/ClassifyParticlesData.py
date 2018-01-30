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
import shutil


# Import keras libraries
from tensorflow.python.keras.preprocessing import image as image_keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# Import local libraries
import CNN_functions


class ClassifyParticlesData(object):

	# Create instance variable since config passed to IrisData instance. 
	def __init__(self, config):
		self.config = config
		self.train_count = 0 
		self.image_train_count = 0 



	def _get_images_per_class(self): 
		classes_dict = defaultdict(int)

		list_of_classes = glob.glob(self.config.train_images_dir + "*")
		list_of_classes.extend(glob.glob(self.config.val_images_dir + "*"))

		for iris_class_path in list_of_classes:
			iris_class_name = iris_class_path.split("/")[-1]

			classes_dict[iris_class_name] +=  len(glob.glob(iris_class_path + "/*"))

		return classes_dict

	def _get_batches_per_epoch(self, directory, batch_size):
		"""
		Args: 
		directory: Root directory for folder (includes sub-directories with all the classes)
		batch_size: The number of files per batch. 
		Returns: 
		Number of epochs needed to iterate through an entire dataset with a given batch_size
		"""
		num_files = self._get_image_count(directory)
		# Need to cast so don't truncate results due to int division. 
		# Use ceil to make sure a single epoch represents a full-cylce of all the images. 
		return math.ceil(num_files/float(batch_size)) 



	def _get_image_count(self, directory):
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



	@staticmethod # Indicates that preprocess_image is a static method (syntactic sugar)
	def preprocess_image_rgb_datasetNorm(x):
		"""
		Description assuming color image.
		Pre-trained VGG16 models expect BGR, so need to return BGR format.
		Args
		x: A RGB image as a numpy tensor (since keras loads through a PIL format)
		Return
		x: A normalized BGR image as a numpy tensor (since VGG assumes BGR formated for pre-loaded weights)
		"""

		# 'RGB'->'BGR' (PIL provided RGB input but need BGR for VGG16 pretrained model)
		x = x[..., ::-1]

		# DATASET AVERAGES (average of each channel)
		# Zero center images based on imagenet 
		# Zero-center by mean pixel of entire dataset (calculated from dataset)
		x[..., 0] -= 90.61598179  # Blue
		x[..., 1] -= 129.97525112 # Green 
		x[..., 2] -= 103.00621832 # Red

		return x

	@staticmethod 
	def preprocess_image_rgb_imageNorm(x):
		"""
		Description assuming color image.
		Pre-trained VGG16 models expect BGR, so need to return BGR format.
		Args
		x: A RGB image as a numpy tensor (since keras loads through a PIL format)
		Return
		x: A normalized BGR image as a numpy tensor (since VGG assumes BGR formated for pre-loaded weights)
		"""

		# 'RGB'->'BGR' (PIL provided RGB input but need BGR for VGG16 pretrained model)
		x = x[..., ::-1]

		# PER-IMAGE NORMALIZATION (average of image)
		# median_intensity = np.median(x)
		x[..., 0] -= median_intensity # Blue
		x[..., 1] -= median_intensity # Green 
		x[..., 2] -= median_intensity # Red

		return x

	@staticmethod 
	def preprocess_image_gray_imageNorm(x):
		"""
		Args
		x: A grayscale image as a numpy tensor 
		Return
		x: A normalized grayscale image as a numpy tensor 
		"""

		# PER-IMAGE GRAYSCALE NORMALIZATION 
		median_intensity = np.median(x)
		x[..., 0] -= median_intensity# Blue

		return x

	def get_preprocess_func(self):

		if (self.config.preprocess_func == "rgb_datasetNorm"):
			return self.preprocess_image_rgb_datasetNorm
		elif (self.config.preprocess_func == "rgb_imageNorm"):
			return self.preprocess_image_rgb_imageNorm
		elif (self.config.preprocess_func == "gray_imageNorm"):
			return self.preprocess_image_gray_imageNorm
		else:
			raise RuntimeError("preprocess_func configuration doesn't exist. Update to existing preprocess function. ")


	def create_training_generator(self, train_dir_path, save_to_dir_bool=False):

		#  If not already, initialize batches per epoch. 
		if (self.config.batches_per_epoch_train == None):
			self.config.batches_per_epoch_train = self._get_batches_per_epoch(train_dir_path, self.config.batch_size)
		print "Batches per Epoch for Training: %d"%(self.config.batches_per_epoch_train)

		# Create directory for augmented images to be saved (if save_to_dir_bool set to True)
		augmented_data_dir = None
		if (save_to_dir_bool):
			datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H-%M-%S')
			augmented_data_dir = (self.config.root_data_dir + "image_data/" + self.config.project_folder + 'augmented_data_' + datestring)
			os.mkdir(augmented_data_dir)

	

		# Create image generator class for training data. 
		train_datagen =  ImageDataGenerator(
			preprocessing_function=self.get_preprocess_func(), # Preprocess function applied to each image before any other transformation. Applies normalization.
			rotation_range=45,  
			width_shift_range=0.07,
			height_shift_range=0.07,
			#shear_range=0.1,
			#zoom_range=0.2,
			horizontal_flip=True, 
			vertical_flip = True, 
			fill_mode = "nearest" #default
		)
		
		# Create image generator. 
		train_generator = train_datagen.flow_from_directory(
			train_dir_path,
			target_size = self.config.target_size,
			batch_size = self.config.batch_size,
			shuffle = True, # default
			color_mode = self.config.color, 
			save_to_dir = augmented_data_dir, 
			save_prefix = "augmented_"
		)

		return train_generator

	def create_validation_generator(self, val_dir_path):

		# If not already, initialize batches per epoch. 
		if (self.config.batches_per_epoch_val == None):
			self.config.batches_per_epoch_val = self._get_batches_per_epoch(val_dir_path, self.config.batch_size)
		print "Batches per Epoch for Validation: %d"%(self.config.batches_per_epoch_val)

		# Create image generator class for training data. 
		test_datagen = ImageDataGenerator(
			preprocessing_function=self.get_preprocess_func(), # Preprocess function applied to each image before any other transformation. Applies normalization. 
		)

		# Create image generator. 
		validation_generator = test_datagen.flow_from_directory(
			val_dir_path,
			target_size = self.config.target_size,
			batch_size = self.config.batch_size,
			shuffle = True, # default
			color_mode = self.config.color 
		)

		return validation_generator

	def create_prediction_generator(self, pred_dir_path):

		# Create image generator class for training data. 
		test_datagen = ImageDataGenerator(
			preprocessing_function=self.get_preprocess_func(), # Preprocess function applied to each image before any other transformation. Applies normalization. 
		)

		# Create image generator. 
		prediction_generator = test_datagen.flow_from_directory(
			pred_dir_path,
			class_mode = None, # No labels are returned (useful for predict_generator and evaluate_generator)
			target_size = self.config.target_size,
			batch_size = self.config.batch_size,
			shuffle = True, # default
			color_mode = self.config.color 
		)

		return prediction_generator


	def print_data_summary(self): 
		self.config.logger.info("###### DATA ######")
		self.config.logger.info("Total Training Data: %d", len(glob.glob(self.config.train_images_dir + '*/*.bmp')))
		self.config.logger.info("Total Validation Data: %d", len(glob.glob(self.config.val_images_dir + '*/*.bmp')))
		self.config.logger.info("Images Per Class: %s", self._get_images_per_class())
		self.config.logger.info("###### DATA ######  \n\n")


	def predict_particle_images(self, model, pred_generator, total_batches=1, output_dir=None, labels_to_class=None): 
		""" 
		Description: Predicts the class for each particle provided by a genrator, up to a certain count. 
		Args
		model: Tensorflow model used to predict particles 
		pred_generator: Generator that produces batches of images to be classified. Generator returned by create_prediction_generator().
		total_images: The number of batches of images that will be processes. 
		output_dir: Directory into which predicted images will be saved. Only save images if directory exists. 
		labels_to_class: Dictionary structure that has labels as keys to the class names. Only include struct if want to save predictions. 
		"""

		# Track overall label metrics
		all_pred = None

		# Setup output directory
		if (output_dir):
			if (not labels_to_class):
				raise RuntimeError("In order to save output images, please provide dictionary for labels_to_class argument.")
			if (os.path.exists(output_dir)): 
				self.delete_folder_with_confirmation(output_dir)	

			os.makedirs(output_dir)
			for label, class_name in labels_to_class.iteritems():
				os.makedirs(output_dir + class_name)


		# Validate across multiple batches
		for _ in range(total_batches):
			img_input =  next(pred_generator)
			label_pred = model.predict_on_batch(img_input)


			self.split_batch_into_class_folders(img_input, label_pred, output_dir, labels_to_class)

			
			# Append to overall metrics
			if all_pred is None: 
				all_pred = label_pred
			else: 
				all_pred = np.append(all_pred, label_pred, axis=0)


		return all_pred

	def split_batch_into_class_folders(self, img_input, label_pred, output_dir, labels_to_class):
		"""
		Description: Takes a batch of images with correspoinding categorigcal/softmax predictions. Uses output_dir and labels_to_class to store outputs. 
		Note: Seperate function to make code more readable. 
		"""
		for index in range(img_input.shape[0]):
			label = np.argmax(label_pred[index])
			datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H-%M-%S-%f')
			file_name = str("%0.04fconfidence_%s"%(label_pred[index][label], datestring))
			img_save_path = output_dir + labels_to_class[label] + "/" + file_name + ".bmp"
			cv2.imwrite(img_save_path, img_input[index])


	def delete_folder_with_confirmation(self, folder_path):
		# User confirmation before deleting folder
		user_question = 'Delete %s: (yes/no): '%(folder_path)
		reply = str(raw_input(user_question)).lower().strip()
		if (reply == 'yes'):
			shutil.rmtree(folder_path)
		else:
			raise RuntimeError("Output folder already exists. However, user indicate not to delete output folder.")



	def validate_epoch(self, model, val_generator): 
		""" 
		Validates model for a single epoch. Provides average results across the entire epoch. 
		"""

		# Track overall label metrics
		all_truth = None
		all_pred = None

		# Validate across multiple batches
		for _ in range(self.config.batches_per_epoch_val):
			img_input, label_truth =  next(val_generator)
			label_pred = model.predict_on_batch(img_input)
			
			# Append to overall metrics
			if all_truth is None: 
				all_truth = label_truth
				all_pred = label_pred
			else: 
				all_truth = np.append(all_truth, label_truth, axis=0)
				all_pred = np.append(all_pred, label_pred, axis=0)


		accuracy =CNN_functions.get_classification_accuracy_perBatch(all_truth, all_pred)
		confusion = CNN_functions.get_confusion_matrix(all_truth, all_pred)

		# Output results
		self.config.logger.info("Validation Results")
		self.config.logger.info("Validation accuracy: %1.3f" %(accuracy))
		self.config.logger.info("Map Class Name to Class Number: %s", val_generator.class_indices)
		self.config.logger.info("Confusion Matrix:")
		self.config.logger.info(confusion)
		self.config.logger.info("\n\n\n")


	def train_epoch(self, model, train_generator, in_house = True):
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
				img_input, label_truth =  next(train_generator)
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
				train_generator, # Produces training data for a single batch. 
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



	def train(self, model, train_generator, val_generator,): 

		for epoch in range(self.config.num_epochs): 

			# Training for single epoch
			self.train_epoch(model, train_generator, in_house = False)


			# Validation for single epoch
			self.validate_epoch(model, val_generator)


	def train_entire_model(self, model, train_generator, val_generator,): 
		# Compiles model
		# Note: categorical_crossentropy requires prediction/truth data to be in categorical format. 
		model.compile(optimizer=self.config.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		model.summary(print_fn = self.config.logger.info)

		count = 0 
		while True:
			count += 1
			self.config.logger.info("######   Entire Model Training  ######")
			self.train(model, train_generator, val_generator,)
			CNN_functions.save_model(model, self.config.weight_file_output + str(count%1) + ".h5", self.config) # Save








