"""
    File name: 
    Author: Sidney Primas
    Date created: 
    Python Version: 2.7
    Description: 
"""


# Base packages 
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import math

# Import OpenCV
import cv2

# Configuration
source_directory = "./data/meeting3/"

# Obtain images from source_directory
file_list = glob(source_directory + "*")

# Iterate through each input file
for ref_input in file_list: 
	im_original = cv2.imread(ref_input, cv2.IMREAD_GRAYSCALE)
	size = im_original.shape

	# Blurred Image
	fig = plt.figure()
	plt.imshow(im_original, cmap='gray', interpolation='nearest')
	plt.title(ref_input)
	plt.axis('off')



plt.show()
