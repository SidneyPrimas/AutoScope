# Particles data

from scipy import ndimage
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import collections
from random import randrange

#ToDo
# Better implementaiton of file selection: Instead of current implementation, another alternative is to create two dictionaries, classes and directories. 
## The directory struct is used to randomly find files, giving equal distribution of all fiels in training. 
## The classes is used to map the directory into it's class, using a dictionary of the following format {dir_path : class_num}

class ParticleSet(object):

	def __init__(self, root_dir, directroy_map, class_size, target_dim, validation_proportion=0.1):
	    self._root_dir = root_dir
	    # The map data structure that relates each directory to a specific class. 
	    self._directroy_map = directroy_map
	    # The number of classes we categorize images into. 
	    self._class_size = class_size
	    # Dimension the input images are upsampled/downsampled to. 
	    self._target_dim = target_dim 
	    # The number of epochs used for training or validation (since this class was instantiated)
	    self._epochs_completed = 0
	    # The number of samples used for training or validation (since this class was instantiated)
	    self._samples_completed = 0

	    # Complete list of files use for training and validation (no overlap between categories)
	    # validation_size: Size of the total dataset used for validation (not used for training).
	    self._trainlist, self._validlist, self._validation_size = self._create_filelists(validation_proportion)

	    print "ParticleSet Initialized"
	    print("Training Set Size: %d"%(len(self._trainlist)))
	    print("Validation Set Size: %d"%(len(self._validlist)))


  	# Note: @Property routes external calls to Class/Object properties through these functions. Allows for elegant constraint checking. 
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
	def epochs_completed(self):
		return self._epochs_completed

	@property
	def samples_completed(self):
		return self._samples_completed

	# Description: Aggregates all filenames into trainlist and validlist
	def _create_filelists(self, validation_proportion): 

		trainlist = []
		validlist = []
		# Build list of all images use for both training and validation. 
		for key in self._directroy_map:
			trainlist.extend(glob(self._root_dir + key + "/*.jpg"))


		# Based on the total number of files in the dataset, determine the valid set. 
		validation_size = int(len(trainlist)*validation_proportion)

		# Move number of original images (validation_size) into validlist. Remove them from trainlist. 
		for i in range(validation_size):
			file_sel = randrange(len(trainlist)) # Returns int between [0 to len(trainlist)-1]
			# Add the selected file to the validation set
			validlist.append(trainlist[file_sel])
			# Remove the file 
			del trainlist[file_sel]



		return trainlist, validlist, validation_size

	# Description: Randomly selects batch_size image from all training data. 
	# Pre-processes image to match the target_dim
	# Organizez batch_size images and labels to be returned. 
	# ToDo: Split into sub-functions
	def next_batch(self, batch_size, validation=False):

		# Select which dataset to use: training or validation
		filelist = self._validlist if validation else self._trainlist

		im_size = self._target_dim * self._target_dim


  		# Initialize numpy arrays to hold data: need to be pre-initialized for efficiency. 
		# data => numpy matrix with [batch, im_size], where images are greyscale
		data = np.empty((batch_size, im_size))
		# labels => one_hot numpy array with [batch, class_size] format
		labels = np.empty((batch_size, self._class_size))

		# Find batch_size images and labels, and place in data and labels list. 
		for n in range(batch_size):
			# Iterate through each sub-section of the class
			file_sel = randrange(len(filelist)) # Returns int between [0 to len(felist)-1]
			file_name = filelist[file_sel]

			### PREPROCESS IMAGE AND APPEND TO DATA STRUCT ###
			# Imresize is just a wrapper around PIL's resize function. 
			# Imresize upsamples and downsamples the image. 
			# Upsampling: Essentially, we upsample by introducing 0 ponits in the matrix, and then use bilinear interpolation. 
			# Downsampling: The risk with downsampling is to get aliasing. In order to remove aliasing, we need to remove high frequency information. 
			# A couple of methods to accomplish this are: 1) Guassian filtering (or blurring) and 2) box averaging (ever 2*2 array)
			# 'bilinear' interpolation takes 4 nearest points, and assumes a continuous transition between them to estimate target point. The bilinear interpolation ensures that we don't alias when downsampling.
			# 'L' is the image mode as defined by PIL, or an 8bit uint (black and white)
			# ToDo: Think about improved implementation that down-samples and upsamples seperately, taking into account aliasing (opencv, scikit)
			temp_im = misc.imresize(misc.imread(file_name), (self._target_dim, self._target_dim), 'bilinear', 'L')
			# Reshape the image into a colum, and insert it into to numpy matrix
			# Append to image data struct
			data[n, :] = np.reshape(temp_im, (1, im_size))


			### CREATE LABELS AND APPEND TO LABEL STRUCT ###
			# Translates from selected directory path to class with directory_map
			end = file_name.rfind("/")
			start = file_name.rfind("/", 0, end)
			key = file_name[start+1:end]
			cur_class = self._directroy_map[key]

			temp_labels = np.zeros((1, self._class_size))
			temp_labels[0][cur_class] = 1
			labels[n, :]= temp_labels

			# Increment sample tracking by 1
			self._samples_completed += 1


		# Increment epoch tracking by 1
		self._epochs_completed += 1
		return data, labels


			


