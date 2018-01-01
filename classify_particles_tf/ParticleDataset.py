from scipy import ndimage
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import collections
from random import randrange
from random import shuffle

class ParticleDataset(object):

	def __init__(self, root_dir, directroy_map, class_size, target_dim, resize = False):
		"""
		_directroy_map: The map data structure that relates each directory to a specific class. 
		_class_size: The number of classes we categorize images into. 
		_target_dim: Dimension the input images are upsampled/downsampled to. 
		"""
		self._root_dir = root_dir
		self._directroy_map = directroy_map
		self._class_size = class_size
		self._target_dim = target_dim 
		self._resize_image = resize

		# Complete list of files use for training and validation (no overlap between categories)
		self._trainlist, self._validlist, self._files_per_class = self._create_filelists_random()
		self._trainlist_class, self._validlist_class = self._create_filelists_class()

		# Sanity checks
		if (self._class_size != len(self._files_per_class)):
			raise Exception("Class size does disagrees in different modules.")

		# Sanity checks
		for key, value in self._files_per_class.iteritems():
			if (value != len(self._trainlist_class[key]) + len(self._validlist_class[key])):
				raise Exception("Images per class don't agree. ")

 
	@property
	def root_dir(self):
		return self._root_dir

	@property
	def directroy_map(self):
		return self._directroy_map

	@property
	def class_size(self):
		return self._class_size

	@property
	def target_dim(self):
		return self._target_dim

	@property
	def files_per_class(self):
		return self._files_per_class

	@property
	def trainlist(self):
		return self._trainlist

	@property
	def validlist(self):
		return self._validlist

	def _create_filelists_class(self):
		trainlist_class = collections.defaultdict(list)
		validlist_class = collections.defaultdict(list)

		# Each class has a list of files that belong to that class. 
		for key, value in self._directroy_map.iteritems():

			trainlist_class[value].extend(glob(self._root_dir + "Training/" + key + "/*.jpg"))
			validlist_class[value].extend(glob(self._root_dir + "Validation/" + key + "/*.jpg"))

		return trainlist_class, validlist_class


	def _create_filelists_random(self): 
		"""
		Description: Aggregates all filenames into trainlist and validlist
		"""

		trainlist = []
		validlist = []
		# Init dictionary to include ints (which are initialized to 0)
		files_per_class = collections.defaultdict(int)

		# Build list of all images use for both training and validation. 
		for key, value in self._directroy_map.iteritems():
			# Create train dataset
			dir_list_train = glob(self._root_dir + "Training/" + key + "/*.jpg")
			trainlist.extend(dir_list_train)

			# Create validation dataset
			dir_list_valid = glob(self._root_dir + "Validation/" + key + "/*.jpg")
			validlist.extend(dir_list_valid)

			files_per_class[value] += len(dir_list_train) + len(dir_list_valid)

		return trainlist, validlist, files_per_class

 
	def _get_image_filename(self, count, validation, per_class_order):
		"""
		Description: Gets file name and class number.
		Random Order: Obtains a validation/training file randomly from all files. 
		Per_Class_Order: Obtains files randomly from within a class, where the class is selected through count.
		"""

		file_name = ""
		class_num = -1
		if(per_class_order): 
			# Select which dataset to use: training or validation
			filestruct = self._validlist_class if validation else self._trainlist_class

			# Determine which class to select from
			class_num = count%self._class_size
			file_sel = randrange(len(filestruct[class_num])) # Returns int between [0 to len(felist)-1]
			file_name = filestruct[class_num][file_sel]

		else:
			filelist = self._validlist if validation else self._trainlist
			file_sel = randrange(len(filelist)) # Returns int between [0 to len(felist)-1]
			file_name = filelist[file_sel]

			# Translates from selected directory path to class with directory_map
			end = file_name.rfind("/")
			start = file_name.rfind("/", 0, end)
			key = file_name[start+1:end]
			class_num = self._directroy_map[key]


		return file_name, class_num


	def next_batch(self, batch_size, validation, per_class_order):
		"""
		Description: Randomly selects batch_size image from all training data. 
		Pre-processes image to match the target_dim
		Organizez batch_size images and labels to be returned. 
		"""

		im_size = self._target_dim * self._target_dim

  		# Initialize numpy arrays to hold data: need to be pre-initialized for efficiency. 
		# data => numpy matrix with [batch, im_size], where images are greyscale
		data = np.empty((batch_size, im_size))
		# labels => one_hot numpy array with [batch, class_size] format
		labels = np.empty((batch_size, self._class_size))

		# Create order of this batch (to ensure not in pre-determined order when sort by class)
		order = range(batch_size)
		shuffle(order)

		# Randomly select the initial class (for per_class_order) so that final classes don't always have less images
		class_start = randrange(self._class_size)

		# Find batch_size images and labels, and place in data and labels list. 
		for count in range(batch_size):

			file_name, class_num = self._get_image_filename(count + class_start, validation, per_class_order)


			# Preprocess image and append to data struct
			temp_im = misc.imread(file_name)
			if (temp_im.shape != (self._target_dim, self._target_dim)):
				if (not self._resize_image):
					raise Exception("All images used for training should have the same dimensions. Look at: %s" % file_name)

				temp_im = misc.imresize(temp_im, (self._target_dim, self._target_dim), 'bilinear', 'L')
				

			# Reshape the image into a colum, and insert it into to numpy matrix
			data[order[count], :] = np.reshape(temp_im, (1, im_size))


			# Create labels and append to label struct
			temp_labels = np.zeros((1, self._class_size))
			temp_labels[0][class_num] = 1
			labels[order[count], :]= temp_labels

		return data, labels


