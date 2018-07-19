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


class SegmentParticlesData(object):

	# Create instance variable since config passed to IrisData instance. 
	def __init__(self, config):
		self.config = config
		self.train_count = 0 
		self.image_train_count = 0 


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


	def get_data_generator(self, image_dir_path, annotations_dir_path):

		# Symmetrically list images and annotations 
		images_list = glob.glob(image_dir_path + "*/*.bmp")
		annotations_list = glob.glob(annotations_dir_path + "*/*.bmp")

		images_list.sort()
		annotations_list.sort()
		reorder =  range(len(images_list))
		random.shuffle(reorder)
		images_list = [images_list[i] for i in reorder]
		annotations_list = [annotations_list[i] for i in reorder]

		# Verify Assumptions: image_list and annotations_list have the same order with the same file names. 
		assert(len(images_list) == len(annotations_list))
		for i in range(len(images_list)):
			image_folder_list = images_list[i].split('/')
			annotation_folder_list = annotations_list[i].split('/')
			assert(image_folder_list[-1] == annotation_folder_list[-1]) # Check image name the same
			assert (image_folder_list[-2] == annotation_folder_list[-2]) # Check coming from same main class folder
		

		#  Returns generator that cycles through the list of images/annotations
		images_iterator = itertools.cycle(images_list)
		annotations_iterator = itertools.cycle(annotations_list)

		# Provide an image and the corresponding annotations. 
		while(True): 

			# Select data gathering method 
			if (self.config.generate_images_with_cropping): # real time cropping
				x_input, y_target = self._get_batch_of_crops(images_iterator, annotations_iterator)
			else:  # images have been pre-cropped
				x_input, y_target = self._get_batch_of_images(images_iterator, annotations_iterator)

			yield np.array(x_input), np.array(y_target)

	def _get_batch_of_crops(self, images_iterator, annotations_iterator):
		"""Gets batches of images/annotations for real time cropping."""

		images_list = []
		annotations_list = []

		# Select # of images equal to the number of main folders. 
		for _ in self.config.input_folders:
			image_path = next(images_iterator)
			annotation_path = next(annotations_iterator)

			image = self._get_image_from_dir(image_path, new_size = self.config.fullscale_target_size)
			label = self._get_annotation_from_dir(annotation_path, new_size = self.config.fullscale_target_size)

			images_list.append(image)
			annotations_list.append(label)


		x_input, y_target = self._get_crops(images_list, annotations_list)

		return x_input, y_target


	def _get_crops(self, images_list, annotations_list):
		"""
		Obtains crops from image and label. Number of crops based on batch_size. 
		"""

		# Create generators
		image_array_iterator = itertools.cycle(images_list)
		annotations_array_iterator = itertools.cycle(annotations_list)

		x_input = []
		y_target = []
		for i in range(self.config.batch_size):
			image = next(image_array_iterator)
			label = next(annotations_array_iterator)

			# Sanity check: the image and labels have the same shape
			assert(image.shape[0] == label.shape[0])
			assert(image.shape[1] == label.shape[1])


			max_height_selection = image.shape[0] - self.config.target_size[0] - 1
			max_width_selection = image.shape[1]  - self.config.target_size[1] - 1

			y1 =  random.randint(0, max_height_selection)
			y2 = y1 + self.config.target_size[0]
			x1 = random.randint(0, max_width_selection)
			x2 = x1 + self.config.target_size[1]

			x_input.append(image[y1:y2, x1:x2])
			label_output_shape = (self.config.target_size[0]*self.config.target_size[1] , self.config.nclasses)
			label_crop_reformat = np.reshape(label[y1:y2, x1:x2], label_output_shape)
			y_target.append(label_crop_reformat)

		return x_input, y_target
		



	def _get_batch_of_images(self, images_iterator, annotations_iterator):

		x_input = []
		y_target = []

		for i in range(self.config.batch_size):

			image_path = next(images_iterator)
			annotation_path = next(annotations_iterator)	

			# Get images
			x_input.append(self._get_image_from_dir(image_path, new_size = self.config.target_size))
			label_output = self._get_annotation_from_dir(annotation_path, new_size = self.config.target_size)
			label_output_shape = (self.config.target_size[0]*self.config.target_size[1] , self.config.nclasses)
			label_output_reshape = np.reshape(label_output, label_output_shape)
			y_target.append(label_output_reshape)

		return x_input, y_target



	def _get_image_from_dir(self, image_path, new_size=None):
		"""
		Loads and resizes image. Zero centers each pixel. Converts to 32-float BGR dataset. 
		"""
		# Use PIL since in correct RGB format. And, Keras relies on PIL. 
		img = PIL.Image.open(image_path) 


		# TODO: Used PIL since VGG16 pretrained network used PIL. Not necessary, and can be changed to numpy implementation. 
		if new_size:
			# PIL loads image as (width, height). My configuration is (height, width)
			new_size_PIL = new_size[::-1] # Move from numpy convention to PIL convention
			if (img.size != new_size_PIL):
				img = img.resize(new_size_PIL, resample = PIL.Image.ANTIALIAS) #PIL implementation: Use antialias to not alias while downsampling. 
		
		x = image_keras.img_to_array(img) # convert to numpy array (as float 32)

		# Zero center images based on imagenet 
		# 'RGB'->'BGR' (PIL provided RGB input)
		x = x[..., ::-1]
		# Zero-center by mean pixel (calculated from dataset)
		x[..., 0] -= self.config.bgr_means[0]  # Blue
		x[..., 1] -= self.config.bgr_means[1] # Green 
		x[..., 2] -= self.config.bgr_means[2] # Red

		return x


	def _get_annotation_from_dir(self, annotation_path, new_size=None):
		"""
		Loads and resizes the annotation. Transforms the annotation into results tensor compatible with a softmax output. 
		"""
		img = PIL.Image.open(annotation_path)  # Use PIL since in correct RGB format. And, Keras relies on PIL.

		# TODO: Used PIL since VGG16 pretrained network used PIL. Not necessary, and can be changed to numpy implementation. 
		if new_size:
			# PIL loads image as (width, height). My configuration is (height, width)
			new_size_PIL = new_size[::-1] # Move from numpy convention to PIL convention
			if (img.size != new_size_PIL):
				#PIL implementation: NEAREST since do not want to blend classes through interpolation. 
				img = img.resize(new_size_PIL, resample = PIL.Image.NEAREST) 


		input_labels = image_keras.img_to_array(img) # convert to numpy array (as float 32)
		input_labels = input_labels[..., ::-1] # 'RGB'->'BGR' (PIL provided RGB input)
		input_labels = input_labels[:,:,0] # Class stored in the B dimension. 
		input_labels_dims = (input_labels.shape[0], input_labels.shape[1])
		
		# Convert to a softmax labeling structure. 
		# Each pixel needs to be labeld with a single class. 
		# E.g.: To label a pixel with 5 classes => [0,0,0,1,0] means pixels is label 4. 
		output_labels = np.zeros(input_labels_dims+ (self.config.nclasses,))
		for c in range(self.config.nclasses):
			output_labels[: , : , c ] = (input_labels == c ).astype(int) # Store 0/1 matrix into single slice of output_labels


		return output_labels


	def _save_validation_images(self, original_img, truth_array, pred_array, base_output_path):
		# orgingal
		path = base_output_path + "_original.jpg"
		cv2.imwrite(path, original_img )
		# truth
		path = base_output_path + "_truth.jpg"
		CNN_functions.save_categorical_aray_as_image(truth_array, path, self.config)
		# prediction 
		path = base_output_path + "_prediction.jpg"
		CNN_functions.save_categorical_aray_as_image(pred_array, path, self.config)



	def print_data_summary(self): 
		self.config.logger.info("###### DATA ######")
		self.config.logger.info("Total Training Data: %d", len(glob.glob(self.config.train_images_dir + '*/*.bmp')))
		self.config.logger.info("Total Validation Data: %d", len(glob.glob(self.config.val_images_dir + '*/*.bmp')))
		self.config.logger.info("###### DATA ######  \n\n")


	def validate_image(self, model, image_path, annotations_path):
		""" 
		Validates the accuracy on a single image. 
		Args
		model: The model used to predict image segmentation
		image_path: string pointing directly to the original image on the disk. 
		annotations_path: string pointing to the annotated image on the disk
		Returns
		label_pred[0]: The predicted segmentation of size (height*width, nclassses). Array with categorical output. 
		label_truth_tensor[0]: The actual segmentation of size (height*width, nclassses). Array with categorical output. 
		"""


		# Get and preprocess label/image
		img_input = self._get_image_from_dir(image_path, new_size = self.config.target_size)
		label_input = self._get_annotation_from_dir(annotations_path, new_size = self.config.target_size)
		
		# Convert label/image to correctly shaped tensor
		img_tensor = np.expand_dims(img_input, axis=0) # Turn single image into tensor (which is what the model expects)
		label_shape = (self.config.target_size[0]*self.config.target_size[1] , self.config.nclasses)
		label_truth = np.reshape(label_input, label_shape)
		label_truth_tensor = np.expand_dims(label_truth, axis=0) 

		# Get model predictions. 
		label_pred = model.predict(img_tensor)


		# Calculate and output optimization accuracy results
		pixel_wise_accuracy_perBatch = CNN_functions.get_pixel_accuracy_perBatch(label_truth_tensor, label_pred)
		self.config.logger.info("Validation Results")
		self.config.logger.info("Validation accuracy: %1.3f" %(pixel_wise_accuracy_perBatch))
		self.config.logger.info("\n\n")

		return label_pred[0], label_truth_tensor[0]


	def predict_image(self, model, image_path):
		""" 
		Predicts the segmented output for an input image.
		Args
		model: The model used to predict image segmentation
		image_path: string pointing directly to the original image on the disk. 
		Returns
		label_pred[0]: The predicted segmentation of size (height*width, nclassses). Array with categorical output. 
		"""


		# Get and preprocess label/image
		img_input = self._get_image_from_dir(image_path, new_size = self.config.target_size)
		
		# Convert label/image to correctly shaped tensor
		img_tensor = np.expand_dims(img_input, axis=0) # Turn single image into tensor (which is what the model expects)
		label_shape = (self.config.target_size[0]*self.config.target_size[1] , self.config.nclasses)

		# Get model predictions. 
		label_pred = model.predict(img_tensor)


		return label_pred[0]



	def validate_epoch(self, model, val_generator, get_particle_accuracy = False): 
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


		pixel_wise_accuracy_perBatch = CNN_functions.get_pixel_accuracy_perBatch(all_truth, all_pred)


		# Save random image from last batch (original, truth, prediction). Once per epoch. 
		if (self.config.debug): 
			random_img_list = range(self.config.batch_size)
			random.shuffle(random_img_list) # Determine which images from the batch to process
			for i in range(2): 
				random_img_list.pop() # removes the last item

				file_prefix = self.config.output_img_dir + str(self.image_train_count) + "_" + str(img_num) + "_"
				# Determine particle_accuracy
				if (get_particle_accuracy): 
					base_save_path = file_prefix + "particle"
					CNN_functions.get_foreground_accuracy_perImage(
						truth_array = label_truth[img_num,:], 
						pred_array = label_pred[img_num,:], 
						config = self.config, 
						radius = self.config.detection_radius,
						base_output_path = base_save_path)

				# Save validation images, including original, ground truth and predictions. 
				base_save_path = file_prefix + "pixel"
				pixel_wise_accuracy_perImage = CNN_functions.get_pixel_accuracy_perImage(label_truth[img_num,:], label_pred[img_num,:])
				# Output results
				self.config.logger.info("Output Image: %s"%(base_save_path))
				self.config.logger.info("Pixel Accuracy: %1.8f"%(pixel_wise_accuracy_perImage))
				self._save_validation_images(
					original_img= img_input[img_num,:,:],
					truth_array= label_truth[img_num,:], 
					pred_array= label_pred[img_num,:], 
					base_output_path= base_save_path)

		# Output results
		self.config.logger.info("Validation Results")
		self.config.logger.info("Validation accuracy: %1.8f" %(pixel_wise_accuracy_perBatch))
		self.config.logger.info("\n\n")

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
			self.validate_epoch(model, val_generator, get_particle_accuracy = True)

	def train_entire_model(self, model, train_generator, val_generator,): 
		# Compiles model
		# Note: categorical_crossentropy requires prediction/truth data to be in categorical format. 
		model.compile(optimizer=self.config.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
		model.summary(print_fn = self.config.logger.info)

		count = 0 
		while True:
			count +=1
			self.config.logger.info("######   Entire Model Training  ######")
			self.train(model, train_generator, val_generator,)
			CNN_functions.save_model(model, self.config.weight_file_output + str(self.image_train_count%1) + ".h5", self.config) # Save







