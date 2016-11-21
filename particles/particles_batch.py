# Particles data

from scipy import ndimage
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import collections
from random import randrange




class ParticleSet(object):

	def __init__(self, root_dir, classes, target_dim):
	    self._root_dir = root_dir
	    self._classes = classes
	    self._target_dim = target_dim
	    self._epochs_completed = 0
	    self._samples_completed = 0

	    print "ParticleSet Initialized"


  	# Note: @Property routes external calls to Class/Object properties through these functions. Allows for elegant constraint checking. 
	@property
	def root_dir(self):
		return self._root_dir

	@property
	def classes(self):
		return self._classes

	@property
	def target_dim(self):
		return self._target_dim

	@property
	def epochs_completed(self):
		return self._epochs_completed

	@property
	def samples_completed(self):
		return self._samples_completed

	# Description: Randomly selects batch_size image from all training data. 
	# Pre-processes image to match the target_dim
	# Organizez batch_size images and labels to be returned. 
	# ToDo: Split into sub-functions
	def next_batch(self, batch_size):

		glob_identifier = "/*.jpg"
		im_size = self._target_dim * self._target_dim


  		# Initialize numpy arrays to hold data: need to be pre-initialized for efficiency. 
		# data => numpy matrix with [batch, im_size], where images are greyscale
		data = np.empty((batch_size, im_size))
		# labels => one_hot numpy array with [batch, len(classes)] format
		labels = np.empty((batch_size, len(self._classes)))

		# Fine batch_size images and labels, and place in data, labels list. 
		for n in range(batch_size):
			# Iterate through each sub-section of the class
			c = randrange(len(self._classes)) # Returns int between [0 to len(classes)-1]
			sub = randrange(len(self._classes[c])) 

			# Extract all files from dessignated folder. 
			# ToDo: Increase efficiency in extracting file names
			filelist = glob(self._root_dir + self._classes[c][sub] + glob_identifier)
			file_sel = randrange(len(filelist))

			### PREPROCESS IMAGE AND APPEND TO DATA STRUCT ###
			# Imresize is just a wrapper around PIL's resize function. 
			# Imresize upsamples and downsamples the image. 
			# Upsampling: Essentially, we upsample by introducing 0 ponits in the matrix, and then use bilinear interpolation. 
			# Downsampling: The risk with downsampling is to get aliasing. In order to remove aliasing, we need to remove high frequency information. 
			# A couple of methods to accomplish this are: 1) Guassian filtering (or blurring) and 2) box averaging (ever 2*2 array)
			# 'bilinear' interpolation takes 4 nearest points, and assumes a continuous transition between them to estimate target point. The bilinear interpolation ensures that we don't alias when downsampling.
			# 'L' is the image mode as defined by PIL, or an 8bit uint (black and white)
			# ToDo: Think about improved implementation that down-samples and upsamples seperately, taking into account aliasing (opencv, scikit)
			temp_im = misc.imresize(misc.imread(filelist[file_sel]), (self._target_dim, self._target_dim), 'bilinear', 'L')
			# Reshape the image into a colum, and insert it into to numpy matrix
			# Append to image data struct
			data[n, :] = np.reshape(temp_im, (1, im_size))


			### CREATE LABELS AND APPEND TO LABEL STRUCT ###
			temp_labels = np.zeros((1, len(self._classes)))
			temp_labels[0][c] = 1
			labels[n, :] = temp_labels
			# Increment sample tracking by 1
			self._samples_completed += 1


		# Increment epoch tracking by 1
		self._epochs_completed += 1
		return data, labels


			


