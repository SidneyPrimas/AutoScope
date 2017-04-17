"""
    File name: resample.py
    Author: Sidney Primas
    Date created: 4/17/2017
    Python Version: 2.7
    Description: Given the reference images of urine particles, we resample all the images to alter the digital resolution of the image. 
"""

# PIL: image processing tool
import PIL # Used to call specific image filters. 
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Function Description: Show original, downsampled, and upsampled image. 
def show_resampled_images(original_im, downsample_im, upsample_im):
	## Interpolation: By default, we interpolate the base pixels in the image when we use image show. 
	# To turn this off, we set the interpolation to 'none'. However, 'none' isn't avaialbe, so we use 'nearest'.
	# The 'nearest' interpolation is equivilant to 'none' at normal scale, but allows for reinterpolation when the image is scaled in a pdf. 
	fig = plt.figure()
	a=fig.add_subplot(1,3,1)
	plt.title("Original")
	imgplot = plt.imshow(original_im, cmap='gray', interpolation='nearest')
	plt.axis('off')
	a=fig.add_subplot(1,3,2)
	plt.title("Downsampled")
	imgplot = plt.imshow(downsample_im, cmap='gray', interpolation='nearest')
	plt.axis('off')

	a=fig.add_subplot(1,3,3)
	plt.title("Upsampled")
	imgplot = plt.imshow(upsample_im, cmap='gray', interpolation='nearest')
	plt.axis('off')
	plt.show()


### Configurable Settings ###
debug_flag = 0
original_resolution = 0.54 # um/pixel
target_resolution = 1.2 #um/pixel
source_data_directory = "./data/IrisDB_resampling/"

## Calculate width reduction
width_reduction = original_resolution/target_resolution
print "Pixel Width/Height Reduction: %f" % (width_reduction)


## Obtain images from all classes (from both validation and training folders)
image_path_list = glob(source_data_directory + "Validation/*/*.jpg")
image_path_list.extend(glob(source_data_directory + "Training/*/*.jpg"));


# Step 1: Identify the image with the most number of pixels. Calculate the final size of that image. 
# Remember: Close images after they have been opened. 
max_width = 0 
for i, path in enumerate(image_path_list):
	with Image.open(path) as im:
		width, height = im.size

		# All reference images should be square images
		if width != height:
			raise Exception("All references images are assumed to be square images. However, they are not. Width: %d, Height: %d." % (width, height))

		# Identify largest height of all images. 
		if height > max_width: 
			max_width = height

print "Maximum Image Size: %dx%d" % (max_width, max_width)

# Calculate the dimensions of the largest image after downsampling. All images will be resampled to this. 
final_dim = int(round(max_width * width_reduction))
print "Output Image Size: %dx%d" % (final_dim, final_dim)

# Step 2: Resample all images by first downsampling to alter um/pixel resolution, and then upsampling to the same image size. 
## Save each altered image in the correct folder within training/validation. 

# Create destination folder
for i, path in enumerate(image_path_list):
	with Image.open(path) as original_im:
		# Calculate the updated width of the image
		width, height = im.size
		target_dim = int(round(width * width_reduction))

		if debug_flag: 
			print "Original Width: %f" % width
			print "Target Dim: %f" % target_dim

		# Downsample the image using PIL directly 
		# ToDO: Figure out the best antialiasing pre-processing. Understand this. Below is a shortcut solution. 
		downsample_im = original_im.resize((target_dim, target_dim), PIL.Image.ANTIALIAS)
		# Upsample the image to ensure all images have uniform dimensions. 
		upsample_im = downsample_im.resize((final_dim, final_dim), PIL.Image.BILINEAR)

		# Show images (for debugging purposes)
		if (debug_flag) and (i == 9500): 
			show_resampled_images(original_im, downsample_im, upsample_im)

		#print path
		upsample_im.save(path)



