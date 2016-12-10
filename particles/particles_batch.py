# Particles data

from scipy import ndimage
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import collections
from random import randrange

class ParticleSet(object):

	def __init__(self, root_dir, directroy_map, class_size, target_dim):
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
	    # TODO: Create structure to hold directory/file info
	    self._trainlist, self._validlist, self._files_per_class = self._create_filelists()
		# Sanity check class_Size
	    if (self._class_size != len(self._files_per_class)):
	    	raise Exception("Class size does disagrees in different modules.")


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

	@property
	def files_per_class(self):
		return self._files_per_class

	@property
	def trainlist(self):
		return self._trainlist

	@property
	def validlist(self):
		return self._validlist


	# Description: Aggregates all filenames into trainlist and validlist
	def _create_filelists(self): 

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

			#TODO: Debug
			# print "Debugging: %d"%(self._samples_completed)
			# print "File_Name: %s"%(file_name)
			# print "Key: %s"%(key)
			# print "Current Class: %d"%(cur_class)
			# print temp_labels
			# print "\n"

			# Increment sample tracking by 1
			self._samples_completed += 1


		# Increment epoch tracking by 1
		self._epochs_completed += 1
		return data, labels


			


