"""
    File name: downsample_demo.py
    Author: Sidney Primas
    Date created: 4/17/2017
    Python Version: 2.7
    Description: Given a source_data_directory, graphs images in 1) original format, 2) downsampled format, and 3) upsampled format. 
"""

# Execution: 
## Adjust the parameters (original_dim, target_dim, source_data_directory and classes). Run. 

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


original_resolution = 1.36
target_resolution = 10.0
final_dim = 52
width_reduction = original_resolution/target_resolution
print "Pixel Width/Height Reduction: %f" % (width_reduction)


source_data_directory = "/Users/sidneyprimas/Documents/Professional/MIT/Sodini/Microscope/codebase/TensorFlow/data/IrisDB_notResampled/Validation/"
save_path = "./data_output/20171026_resample/"

## Obtain images from select classes  
classes = {
	#0: "IRIS-BACT", 
	#0: "IRIS-RBC",
	#2: "IRIS-SPRM",
	0: "IRIS-WBC",
	1: "IRIS-CLUMP-WBCC",
	#4: "IRIS-SQEP",
}


# Iterate through each selected class
for c in range(len(classes)):
	target_file = source_data_directory + classes[c] + "/0.jpg"


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
	width, _ = original_im.size
	target_dim = int(round(width * width_reduction))
	downsample_im = original_im.resize((target_dim, target_dim), PIL.Image.ANTIALIAS)
	upsample_im = downsample_im.resize((final_dim, final_dim), PIL.Image.BILINEAR)

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

	original_im.save(save_path+classes[c]+"_"+str(int(target_resolution))+"_"+"original", "JPEG", quality=100)
	downsample_im.save(save_path+classes[c]+"_"+str(int(target_resolution))+"_"+"downsample", "JPEG", quality=100)
	upsample_im.save(save_path+classes[c]+"_"+str(int(target_resolution))+"_"+"resampledTo_"+str(final_dim)+"px", "JPEG", quality=100)

	#misc.imsave('face.png', f)




plt.show()