# Particles data

# Scipy image processing tool
from scipy import ndimage
from scipy import misc

# PIL image processing tool
import PIL
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

original_dim = 128
target_dim = 26
source_data_directory = "/Users/sidneyprimas/Documents/Professional/MIT/Sodini/Microscope/TensorFlow/data/IrisDB_resampling/"
import_count = 2

## Obtain images from all classes
#dir_input_data = glob(source_data_directory + "Validation/*")

## Obtain images from select classes  
classes = {
	0: "IRIS-BACT", 
	1: "IRIS-RBC",
	2: "IRIS-SPRM",
	3: "IRIS-WBC",
	#4: "IRIS-SQEP",
}

# Iterate through each selected class
for c in range(len(classes)):
	target_file = source_data_directory + classes[c] + "/2.jpg"


	### Downsample the Image Using Scipy (wrapper around PIL) ###
	# Imresize is just a wrapper around PIL's resize function. 
	# Imresize upsamples and downsamples the image. 
	# Upsampling: Essentially, we upsample by introducing 0 ponits in the matrix, and then use bilinear interpolation. 
	# Downsampling: The risk with downsampling is to get aliasing. In order to remove aliasing, we need to remove high frequency information. 
	# A couple of methods to accomplish this are: 1) Guassian filtering (or blurring) and 2) box averaging (ever 2*2 array)
	# 'bilinear' interpolation takes 4 nearest points, and assumes a continuous transition between them to estimate target point. The bilinear interpolation ensures that we don't alias when downsampling.
	# 'L' is the image mode as defined by PIL, or an 8bit uint (black and white)
	# ToDo: Think about improved implementation that down-samples and upsamples seperately, taking into account aliasing (opencv, scikit)
	

	### Downsample the Image Using PIL Directly ###
	# Shortcut solution to pre-filtering an image to reduce aliasing. 
	# ToDO: Need proper implementation that actually solves aliasing
	original_im = Image.open(target_file)
	downsample_im = original_im.resize((target_dim, target_dim), PIL.Image.ANTIALIAS)
	upsample_im = downsample_im.resize((original_dim, original_dim), PIL.Image.BILINEAR)

	## Interpolation: By default, we interpolate the base pixels in the image when we use image show. 
	# To turn this off, we set the interpolation to 'none'. However, 'none' isn't avaialbe, so we use 'nearest'.
	# The 'nearest' interpolation is equivilant to 'none' at normal scale, but allows for reinterpolation when the image is scaled in a pdf. 
	fig = plt.figure()
	a=fig.add_subplot(1,3,1)
	plt.title("Original: " + classes[c])
	imgplot = plt.imshow(original_im, cmap='gray', interpolation='nearest')
	plt.axis('off')
	a=fig.add_subplot(1,3,2)
	plt.title("Downsampled: " + classes[c])
	imgplot = plt.imshow(downsample_im, cmap='gray', interpolation='nearest')
	plt.axis('off')

	a=fig.add_subplot(1,3,3)
	plt.title("Upsampled: " + classes[c])
	imgplot = plt.imshow(upsample_im, cmap='gray', interpolation='nearest')
	plt.axis('off')

	#misc.imsave('face.png', f)




plt.show()