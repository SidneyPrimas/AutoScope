# Particles data

from scipy import ndimage
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import collections

import tensorflow as tf

# With the IrisDB dataset, we have 10 classes of particles. . 
NUM_CLASSES = 10
target_dim = 52
target_size = 52*52
base_data_directory = "/Users/sidneyprimas/Documents/Professional/MIT/Sodini/Microscope/IrisDB/"
glob_identifier = "/*.jpg"

# classes => dictionary with {class_num : list of directory names}
classes = {
	0: ["IRIS-BACT"], 
	1: ["IRIS-CLUMPS/IRIS-WBCC", "IRIS-CLUMPS/IRIS-YSTS"], 
	2: ["IRIS-CRYST/IRIS-CAOX", "IRIS-CRYST/IRIS-CAPH", "IRIS-CRYST/IRIS-TPO4", "IRIS-CRYST/IRIS-URIC"], 
	3: ["IRIS-HYAL"], 
	4: ["IRIS-NHYAL/IRIS-CELL", "IRIS-NHYAL/IRIS-GRAN"], 
	5: ["IRIS-NSQEP/IRIS-REEP", "IRIS-NSQEP/IRIS-TREP"],
	6: ["IRIS-RBC"],
	7: ["IRIS-SPRM"],
	8: ["IRIS-SQEP"],
	9: ["IRIS-WBC"],
}

# Initialize data structures to map to lists. 
# data => dictionary with {class_num : list of numpy_array images}
# images => 2D images with a single color channel (greyscale)
data = collections.defaultdict(list)
# labels => dictionary with {class_num : one_hot list in [10,1] format}
labels = collections.defaultdict(list)

# Import 5 images in each category. 
import_count = 5

# Iterate through each class. 
for c in range(0,NUM_CLASSES):
	# Iterate through each sub-section of the class
	for i in range(len(classes[c])):
		print classes[c][i]
		filelist = glob(base_data_directory + classes[c][i] + glob_identifier)
		for n in range(import_count):
			print filelist[n]

			### PREPROCESS IMAGE AND APPEND TO DATA STRUCT ###
			# Imresize is just a wrapper around PIL's resize function. 
			# Imresize upsamples and downsamples the image. 
			# Upsampling: Essentially, we upsample by introducing 0 ponits in the matrix, and then use bilinear interpolation. 
			# Downsampling: The risk with downsampling is to get aliasing. In order to remove aliasing, we need to remove high frequency information. 
			# A couple of methods to accomplish this are: 1) Guassian filtering (or blurring) and 2) box averaging (ever 2*2 array)
			# 'bilinear' interpolation takes 4 nearest points, and assumes a continuous transition between them to estimate target point. The bilinear interpolation ensures that we don't alias when downsampling.
			# 'L' is the image mode as defined by PIL, or an 8bit uint (black and white)
			# ToDo: Think about improved implementation that down-samples and upsamples seperately, taking into account aliasing (opencv, scikit)
			temp_im = misc.imresize(misc.imread(filelist[n], ), (52,52), 'bilinear', 'L')
			# Append to image data struct
			data[c].append(temp_im)

			### CREATE LABELS AND APPEND TO LABEL STRUCT ###
			temp_labels = np.zeros((len(classes), 1))
			temp_labels[c] = 1
			labels[c].append(temp_labels)
		plt.figure(c)
		plt.imshow(data[c][0])
		plt.title(classes[c][0])



plt.show()
			



# ToDo: Thinking about using np.memmap