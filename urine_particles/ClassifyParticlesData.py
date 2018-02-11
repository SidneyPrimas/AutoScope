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
import re


# Import keras libraries
from tensorflow.python.keras.preprocessing import image as image_keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, flip_axis, random_shift, random_rotation

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

		x /= 128.0 # Normalize Results in range from [-1,1].
		#x /= np.std(x)

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
		median_intensity = np.median(x)
		x[..., 0] -= median_intensity # Blue
		x[..., 1] -= median_intensity # Green 
		x[..., 2] -= median_intensity # Red

		x /= 128.0 # Normalize. Results in range from [-1,1].
		#x /= np.std(x)

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
		x[..., 0] -= median_intensity # Blue

		x /= 128.0 # Normalize. Results in range from [-1,1].
		#x /= np.std(x)

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
		"""
		Description: Creates a custom generator that returns model inputs and labels. 
		Use keras library. 
		Current advantages of using keras: Additional augmentation functions (shift and rotation) and saving augmented data to dir automatically. 
		Current disadvantages: No option imlemented for including centroids in input data. 
		"""

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
			width_shift_range=0.2,
			height_shift_range=0.2,
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

		# Ensure that mapping from class name to label is consisent. 
		if (self.config.class_mapping != train_generator.class_indices):
			raise ValueError("class_mapping in class configuration doesn't match with class folder structure. Please reconcile.")

		return train_generator

	def create_validation_generator(self, val_dir_path):
		"""
		Description: Creates a custom generator that returns model inputs and labels. 
		Use keras library. 
		"""

		# If not already, initialize batches per epoch. 
		if (self.config.batches_per_epoch_val == None):
			self.config.batches_per_epoch_val = self._get_batches_per_epoch(val_dir_path, self.config.batch_size)
		print "Batches per Epoch for Validation: %d"%(self.config.batches_per_epoch_val)


		# Create image generator class for training data. 
		# Note: Since we have duplicates to balance classes, still apply horizontal + vertical flips so don't use identical images. 
		test_datagen = ImageDataGenerator(
			preprocessing_function=self.get_preprocess_func(), # Preprocess function applied to each image before any other transformation. Applies normalization. 
			horizontal_flip=True, 
			vertical_flip = True, 
		)

		# Create image generator. 
		validation_generator = test_datagen.flow_from_directory(
			val_dir_path,
			target_size = self.config.target_size,
			batch_size = self.config.batch_size,
			shuffle = True, # default
			color_mode = self.config.color 
		)

		# Ensure that mapping from class name to label is consisent. 
		if (self.config.class_mapping != validation_generator.class_indices):
			raise ValueError("class_mapping in class configuration doesn't match with class folder structure. Please reconcile.")

		return validation_generator


	def create_custom_labeled_generator(self, target_directory, augment_data): 
		"""
		Description: Creates generator that produces the models input data and their corresponding labels. 
		Return
		x_input: Numpy array that includes the images in the folders within pred_dir_path. 
			Depending on enable_custom_features in config might also include image cnetroids. . 
		y_labels: The corresponding labels to the x_input
		"""

		# If not already, initialize batches per epoch. 
		if (self.config.batches_per_epoch_val == None):
			self.config.batches_per_epoch_val = self._get_batches_per_epoch(val_dir_path, self.config.batch_size)
		print "Batches per Epoch for Validation: %d"%(self.config.batches_per_epoch_val)

		# Setup list of training images.
		images_list = glob.glob(target_directory + '*/*.bmp') # path to training data
		
		# Verify class mapping dictionary
		# Specifically, verify that the auto-generated mapping is identical to the config mapping
		class_path_list = glob.glob(target_directory + '*') # paths to class folders
		class_path_list.sort()
		class_mapping_fromFolder = {}
		for key, class_folder_path in enumerate(class_path_list):
			class_name = class_folder_path.split('/')[-1]
			class_mapping_fromFolder[class_name] = key

		if (self.config.class_mapping != class_mapping_fromFolder):
			raise ValueError("class_mapping in class configuration doesn't match with class folder structure. Please reconcile.")


		# Shuffle image list
		random.shuffle(images_list)

		#  Returns generator that cycles through the list of images
		images_iterator = itertools.cycle(images_list)

		# Provide an image and the corresponding path. 
		while(True): 
			x_inputs, y_labels, _ = self._get_batch_of_images(images_iterator, 
				include_labels=True, 
				include_custom_features=self.config.enable_custom_features,
				augment_data=augment_data)


			yield x_inputs, y_labels


	def create_custom_prediction_generator(self, pred_dir_path):
		"""
		Description: Creates a generator that returns images to be predicted. 
		Return
		x_inputs: Numpy array that includes the images in the folders within pred_dir_path + the centroids of the corresponding image. 
		path_list: Python list that includes the path to the images in x_input
		"""
		# Symmetrically list images
		images_list = glob.glob(pred_dir_path + "*/*.bmp")

		# Shuffle lists
		random.shuffle(images_list)

		#  Returns generator that cycles through the list of images
		images_iterator = itertools.cycle(images_list)

		# Provide an image and the corresponding path. 
		while(True): 
			# Get batch of inputs and their corresponding path lists. 
			x_inputs, _, path_list = self._get_batch_of_images(
				images_iterator, 
				include_labels=False, 
				include_custom_features = self.config.enable_custom_features,
				augment_data=False)
			

			yield x_inputs, path_list


	def _get_batch_of_images(self, images_iterator, include_labels, include_custom_features, augment_data):
		"""
		Description: Get a batch of images with the corresponding 1) labels and 2) load paths for each image. 
		Args: 
		images_iterator: Iterator that yiels a single path to a single input image
		include_labels: Bool that indicates if labels should be generated for this batch. 
		include_custom_features: Bool that indicates if centroid feature should be included in x_inputs
		augment_data: Bool that indicates if real time data augmentation should be applied
		"""
		image_input = []
		coordinates_input = []
		label_list = []
		path_list = []

		for i in range(self.config.batch_size):

			# Get image path 
			image_path = next(images_iterator)
			path_list.append(image_path)

			# Get image
			image_input.append(self._get_image_from_dir(image_path, augment_data, new_size = self.config.target_size))

			# Get labels (if include_labels is True)
			if (include_labels):
				label_list.append(self._get_label_from_imgPath(image_path))

			# Get image cordinates
			if (include_custom_features):
				crop_filename = image_path[image_path.rfind('/')+1:] # Identify image_path file name
				coordinate_features = self._get_custom_features(crop_filename)
				coordinates_input.append(coordinate_features)

		# Build + format objects to be returned
		# Create x_inputs
		if (include_custom_features):
			x_inputs = {'image_input': np.array(image_input), 'features_input': np.array(coordinates_input, dtype=np.float32)}
		else: 
			x_inputs = np.array(image_input)

		# Create y_labels
		if (include_labels):
			y_labels = np.array(label_list)
		else:
			y_labels = None

		return x_inputs, y_labels, path_list


	def _get_custom_features(self, crop_filename):
		"""
		crop_filename: Extract, calculate and normalize the custom features. Specifically, get position information. 
		Args
		crop_filename: The filename of the cropped image that contains coordinates embedded in the name. 
		+ Currently, the standard format is 'img1_43_235.bmp' or 'img1_43_235_cpy20.bmp' or '10um_particle_40_1455_cpy304.bmp'
		image_dim: A tuple that indicates the dimensions of an image. The tuple is given in (height, width) format.
		Return (given in (width, height))
		normalized_pos_list = [coordinate_pos + delta_pos + angular_pos]
		coordinate_pos (list): The raw coordinate position with the origin at pixel (0,0)
		delta_pos (list): The coordinates with the origin at the center of the image
		angular_pos (list): The angular position with the origin at the center of the image
		"""

		# Determine raw coordinate_pos (already returned as float)
		coordinate_pos = CNN_functions.get_coordinates_from_cropname(crop_filename)

		# Swith from (height, width) to (width, height)
		dims_w_h = [float(self.config.canvas_dims[1]), float(self.config.canvas_dims[0])]

		# Calculate delta_pos
		dims_half_w_h = [dims_w_h[0]/2.0, dims_w_h[1]/2.0]
		delta_x = (coordinate_pos[0]-dims_half_w_h[0])
		delta_y = coordinate_pos[1]-dims_half_w_h[1]


		# Determine angular_pos
		angle = math.atan2(delta_y, delta_x)
		mag = math.sqrt(math.pow(delta_y,2) + math.pow(delta_x, 2))
		mag_max = math.sqrt(math.pow(dims_half_w_h[1],2) + math.pow(dims_half_w_h[0], 2))
		

		# Normalize position values
		coordinate_pos = [coordinate_pos[0]/dims_w_h[0], coordinate_pos[1]/dims_w_h[1]]
		delta_pos = [delta_x/dims_half_w_h[0], delta_y/dims_half_w_h[1]]
		angular_pos = [mag/mag_max, angle/math.pi] 

		# Create output list
		normalized_pos_list = coordinate_pos + delta_pos + angular_pos

		return normalized_pos_list


	def _get_label_from_imgPath(self, image_path):
		"""
		Description: Gets a sparse categorical label array from the image_path. 
		+ Obtains the class name from the image path
		+ Uses config.class_mapping to go from class name to label. 
		+ Constructs a sparse categorical label with all 0s except a 1 at label position. 
		"""

		class_name = image_path.split('/')[-2]
		label = self.config.class_mapping[class_name]
		label_array = np.zeros(len(self.config.class_mapping))
		label_array[label] = 1

		return label_array


	def _get_image_from_dir(self, image_path, augment_data, new_size=None):
		"""
		Descriptions: Loads and preprocess image (including resizes,  image norm for each image and augmenting the image). 
		"""
		# Use PIL since in correct RGB format. And, Keras relies on PIL. 
		img = PIL.Image.open(image_path)


		# Convert to grayscale if necessary. Keras implementation also converts from rgb/grayscale first. 
		if ('grayscale' == self.config.color):
			 img = img.convert('L')

		# TODO: Used PIL since VGG16 pretrained network used PIL. Not necessary, and can be changed to numpy implementation. 
		if new_size:
			# PIL loads image as (width, height). My configuration is (height, width)
			new_size_PIL = new_size[::-1] # Move from numpy convention to PIL convention
			if (img.size != new_size_PIL):
				img = img.resize(new_size_PIL, resample = PIL.Image.ANTIALIAS) #PIL implementation: Use antialias to not alias while downsampling. 
		

		x = image_keras.img_to_array(img) # convert to numpy array (as float 32)

		preprocess = self.get_preprocess_func()
		x = preprocess(x)

		# augment image
		x = self._transform_image(x, augment_data)
		
		return x

	def _transform_image(self, x, augment_data):
		"""
		Description: Augments an input image randomly, in real-time. 
		Note: We always flip images around axis since sometimes we have duplicate images in a dataset (since we might pad classes to equalize them)
		ToDo: To improve efficiency, combine rotation/shift into single homography transformation (instead of applying two back to back.)
		Currently, apply two transformations/interpolations sequetially. Can concatenate this into a single transformation. 
		"""

		# Probabilistic horizontal flip
		if np.random.random() < 0.5:
			x = flip_axis(x, 1) # flip around columns

		# Probabilist vertical flip
		if np.random.random() < 0.5:
			x = flip_axis(x, 0) # flip around rows
		
		# Only apply when agumenting data. 
		if (augment_data):
			x = random_rotation(x, 
				rg=45, 
				row_axis=0, 
				col_axis=1, 
				channel_axis=2,
				fill_mode='nearest')

			x = random_shift(x, 
				wrg=0.2, 
				hrg=0.2, 
				row_axis=0, 
				col_axis=1, 
				channel_axis=2,
				fill_mode='nearest')


		return x


	def print_data_summary(self): 
		self.config.logger.info("###### DATA ######")
		self.config.logger.info("Total Training Data: %d", len(glob.glob(self.config.train_images_dir + '*/*.bmp')))
		self.config.logger.info("Total Validation Data: %d", len(glob.glob(self.config.val_images_dir + '*/*.bmp')))
		self.config.logger.info("Images Per Class: %s", self._get_images_per_class())
		self.config.logger.info("###### DATA ######  \n\n")


	def predict_particle_images(self, model, pred_generator, total_batches=1): 
		""" 
		Description: Predicts the class for each particle provided by a genrator, up to a certain count. 
		Args
		model: Tensorflow model used to predict particles 
		pred_generator: Generator that produces batches of images to be classified. Generator returned by create_custom_prediction_generator().
		total_batches: The number of batches of images that will be processes. 
		"""

		# Track overall label metrics
		all_pred = None
		all_path_list = None


		# Validate across multiple batches
		for _ in range(total_batches):
			x_inputs, path_list =  next(pred_generator)	
			label_pred = model.predict_on_batch(x_inputs)

			# Append to overall metrics
			if all_pred is None: 
				all_pred = label_pred
				all_path_list = path_list
			else: 
				all_pred = np.append(all_pred, label_pred, axis=0)
				all_path_list = np.append(all_path_list, path_list, axis=0)

		return all_pred, all_path_list



	def validate_epoch(self, model, val_generator): 
		""" 
		Validates model for a single epoch. Provides average results across the entire epoch. 
		"""

		# Track overall label metrics
		all_truth = None
		all_pred = None

		# Validate across multiple batches
		for _ in range(self.config.batches_per_epoch_val):
			x_inputs, label_truth =  next(val_generator)
			label_pred = model.predict_on_batch(x_inputs)

			
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
		self.config.logger.info("Map Class Name to Class Number: %s", self.config.class_mapping)
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
				x_inputs, label_truth =  next(train_generator)
				metric_output = model.train_on_batch(x_inputs, label_truth)
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








