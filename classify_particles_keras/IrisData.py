# Import basic libraries
import numpy as np
import matplotlib.pyplot as plt
import PIL
import glob
import os
import math
from datetime import datetime
from collections import defaultdict


# Import keras libraries
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# Import local libraries
import CNN_functions


class IrisData(object):

	# Create instance variable since config passed to IrisData instance. 
	def __init__(self, config):
		self.config = config
		self.train_count = 0 
		self.image_train_count = 0 

		# Setup root Keras directory if it doesn't exist. Get user confirmation. 
		self._setup_data_directory()


		# Create generators 
		# Create data generators
		self.train_generator = self._create_training_generator(save_to_dir_bool = self.config.save_to_dir_bool)
		self.val_generator = self._create_validation_generator()



	def _get_images_per_class(self): 
		classes_dict = defaultdict(int)

		list_of_classes = glob.glob(self.config.train_dir + "*")
		list_of_classes.extend(glob.glob(self.config.val_dir + "*"))

		for iris_class_path in list_of_classes:
			iris_class_name = iris_class_path.split("/")[-1]

			classes_dict[iris_class_name] +=  len(glob.glob(iris_class_path + "/*"))
		return classes_dict


	def _setup_data_directory(self):

		if (not os.path.isdir(self.config.root_dir)): #Create Directory
			user_input = raw_input('Do you want to create a new Keras root directory with images? Type yes to continue. \n')
			if (user_input == "yes"): 
				self._create_keras_directory(self.config.original_data_dir)
				self.config.classes = len(glob.glob(self.config.train_dir + "*")) # Update number of classes
			else: 
				raise OSError("Local error in IrisData/__init__(): The keras directory doesn't exist, and user doesn't want to make it.")


	def _get_img_count(self, directory):
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

	def _import_file_to_folder(self, source, destination): 
		img = PIL.Image.open(source) # Import image using PIL
		img = img.resize(self.config.target_size, resample = PIL.Image.NEAREST) #PIL implementation: Use nearest since interpolation doesn't create more data. 
		img = img.convert("RGB") # Converts the greyscale image to an RGB image. 
		img.save(destination, format="JPEG")


	def _create_keras_directory(self, original_data_dir): 

		# Raise an exception if the root_dir already exists
		if (os.path.isdir(self.config.root_dir)): 
			raise OSError("Local error in IrisData/create_keras_directory(): The root_dir already exists.")

		# Make Validation/Training directories 
		os.mkdir(self.config.root_dir)
		os.mkdir(self.config.val_dir)
		os.mkdir(self.config.train_dir)


		# Process each folder in original data directory. 
		for cell_type, class_name in self.config.directory_map.iteritems():

			# Make class directory if they don't exist (in both training and validation folder)
			if (not os.path.isdir(self.config.val_dir +  class_name)): 
				os.mkdir(self.config.val_dir +  class_name)
				os.mkdir(self.config.train_dir +  class_name)

			# List of files in original cell type folders. 
			image_list = glob.glob(original_data_dir + cell_type + "/*.jpg") 
			validation_size = math.ceil(len(image_list)*self.config.validation_percent)
			# Copy each image from folder in original_data_dir to it's class directory in validation or training
			for image_num, original_file_path in enumerate(image_list): 

				# Move images to VALIDATION Folder
				if image_num < validation_size:
					destination_for_file = (self.config.val_dir +  class_name + "/" + cell_type + "_" + str(image_num) + ".jpg")
					self._import_file_to_folder(original_file_path, destination_for_file)
				
				# Move images to TRAINING Folder	
				else:
					destination_for_file = (self.config.train_dir +  class_name + "/" + cell_type + "_" + str(image_num) + ".jpg")
					self._import_file_to_folder(original_file_path, destination_for_file)

	@staticmethod
	def preprocess_image(x):
		"""
		Pre-trained VGG16 models expect BGR, so need to return BGR format. Zero centers each pixel.
		Args
		x: A RGB image as a numpy tensor (since keras loads through a PIL format)
		Return
		x: A BGR image as a numpy tensor (since VGG assumes BGR formated for pre-loaded weights)
		"""

		# Zero center images based on imagenet 
		# 'RGB'->'BGR' (PIL provided RGB input)
		x = x[..., ::-1]
		# Zero-center by mean pixel (based on VGG16 means)
		x[..., 0] -= 103.939 # Blue
		x[..., 1] -= 116.779 # Green
		x[..., 2] -= 123.68 # Red

		return x					


	def _create_training_generator(self, save_to_dir_bool=False):

		#  If not already, initialize batches per epoch. 
		if (self.config.batches_per_epoch_train == None):
			self.config.batches_per_epoch_train = self._get_batches_per_epoch(config.train_dir, self.config.batch_size)
		print "Batches per Epoch for Training: %d"%(self.config.batches_per_epoch_train)

		# Create directory for augmented images to be saved (if save_to_dir_bool set to True)
		augmented_data_dir = None
		if (save_to_dir_bool):
			datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H-%M-%S')
			augmented_data_dir = (self.config.root_dir + 'augmented_data_' + datestring)
			os.mkdir(augmented_data_dir)

	
		# Create image generator class for training data. 
		train_datagen =  ImageDataGenerator(
			preprocessing_function=self.preprocess_image, # Preprocess function applied to each image before any other transformation. Applies normalization.
			rotation_range=45,  
			#width_shift_range=0.1,
			#height_shift_range=0.1,
			#shear_range=0.1,
			#zoom_range=0.2,
			horizontal_flip=True, 
			vertical_flip = True, 
			fill_mode = "nearest" #default
		)
		
		# Create image generator. 
		train_generator = train_datagen.flow_from_directory(
			self.config.train_dir,
			target_size = self.config.target_size,
			batch_size = self.config.batch_size,
			shuffle = True, # default
			color_mode = "rgb", # Important: We upconverted the original grayscale images to RGB. 
			save_to_dir = augmented_data_dir, 
			save_prefix = "augmented_"
		)

		return train_generator

	def _create_validation_generator(self):

		# If not already, initialize batches per epoch. 
		#  Initialized to ENSURE
		if (self.config.batches_per_epoch_val == None):
			self.config.batches_per_epoch_val = self._get_batches_per_epoch(self.config.val_dir, self.config.batch_size)
		print "Batches per Epoch for Validation: %d"%(self.config.batches_per_epoch_val)

		# Create image generator class for training data. 
		test_datagen = ImageDataGenerator(
			preprocessing_function=self.preprocess_image, # Preprocess function applied to each image before any other transformation. Applies normalization. 
		)

		# Create image generator. 
		validation_generator = test_datagen.flow_from_directory(
			self.config.val_dir,
			target_size = self.config.target_size,
			batch_size = self.config.batch_size,
			shuffle = True, # default
			color_mode = "rgb" # Important: We upconverted the original grayscale images to RGB. 
		)

		return validation_generator
	

	def print_data_summary(self): 
		self.config.logger.info("###### DATA ######")
		self.config.logger.info("Total Training Data: %d", self._get_img_count(self.config.train_dir))
		self.config.logger.info("Total Validation Data: %d", self._get_img_count(self.config.val_dir))
		self.config.logger.info("Images Per Class: %s", self._get_images_per_class())
		self.config.logger.info("Map Class Name to Class Number: %s", self.train_generator.class_indices)
		self.config.logger.info("###### DATA ######  \n\n")

	def validate_epoch(self, model): 
		""" 
		Validates model for a single epoch. Provides average results across the entire epoch. 
		"""

		# Track overall label metrics
		all_truth = None
		all_pred = None

		# Validate across multiple batches
		for batch in range(self.config.batches_per_epoch_val):
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
		confusion = CNN_functions.get_confusion_matrix(all_truth, all_pred)


		# Output results
		self.config.logger.info("Validation Results")
		self.config.logger.info("Training accuracy: %1.3f" %(accuracy))
		self.config.logger.info("Confusion Matrix:")
		self.config.logger.info(confusion)
		self.config.logger.info("\n\n\n")

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
				self.train_generator, # Calling the function once produced a tuple with (inputs, targets, weights). Produces training data for a single batch. 
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



	def train(self, model): 

		for epoch in range(self.config.num_epochs): 

			# Training for single epoch
			self.train_epoch(model, in_house = False)

			# Validation for single epoch
			self.validate_epoch(model)
	

	# Possible: move to generic CNN_functions.py. 
	def fine_tune_train(self, model): 

		# Compiles Transfer Learning Model
		CNN_functions.freeze_lower_layers(model, self.config.tl_freeze_layer) # Freeze Lower Layers
		model.compile(optimizer=self.config.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		model.summary(print_fn = self.config.logger.info)
		self.train(model)
		CNN_functions.save_model(model, self.config.tl_model_file, self.config) # Save


		# Perform Fine Tuning Training
		CNN_functions.freeze_lower_layers(model, self.config.ft_freeze_layer) # Freeze Lower Layers
		model.compile(optimizer=self.config.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		model.summary(print_fn = self.config.logger.info)

		while True:
			self.config.logger.info("######   Initiate Fine Tuning Session  ######")
			self.train(model)
			CNN_functions.save_model(model, self.config.ft_model_file, self.config) # Save


	def transfer_learn_train(self, model):

		# Compiles model
		CNN_functions.freeze_lower_layers(model, self.config.tl_freeze_layer) # Freeze Lower Layers
		model.compile(optimizer=self.config.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		model.summary(print_fn = self.config.logger.info)

		while True:
			self.config.logger.info("######   Initiate Transfer Learning Session  ######")
			self.train(model)
			CNN_functions.save_model(model, self.config.tl_model_file, self.config) # Save

	def train_entire_model(self, model): 
		# Compiles model
		model.compile(optimizer=self.config.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		model.summary(print_fn = self.config.logger.info)

		while True:
			self.config.logger.info("######   Entire Model Training  ######")
			self.train(model)
			CNN_functions.save_model(model, self.config.tl_model_file, self.config) # Save








