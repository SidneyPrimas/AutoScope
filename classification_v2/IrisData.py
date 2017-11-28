# Import basic libraries
import numpy as np
import matplotlib.pyplot as plt
import PIL
import glob
import os
import math
from datetime import datetime
from collections import defaultdict


#Import keras libraries
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


class IrisData(object):

	# Create instance variable since config passed to IrisData instance. 
	def __init__(self, config):
		self.config = config

		# Setup root Keras directory if it doesn't exist. Get user confirmation. 
		self._setup_data_directory()



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


	def create_training_generator(self, save_to_dir_bool=False):

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
			preprocessing_function=None, # Preprocess function applied to each image before any other transformation. Applies normalization.
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

	def create_validation_generator(self):

		# If not already, initialize batches per epoch. 
		#  Initialized to ENSURE
		if (self.config.batches_per_epoch_val == None):
			self.config.batches_per_epoch_val = self._get_batches_per_epoch(self.config.val_dir, self.config.batch_size)
		print "Batches per Epoch for Validation: %d"%(self.config.batches_per_epoch_val)

		# Create image generator class for training data. 
		test_datagen = ImageDataGenerator(
			preprocessing_function=None, # Preprocess function applied to each image before any other transformation. Applies normalization. 
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
		self.config.logger.info("###### DATA ######  \n\n")

