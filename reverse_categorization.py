"""
    File name: reverse_categorization.py
    Author: Sidney Primas
    Date created: 06/27/2017
    Python Version: 2.7
    Description: Perform segmentation and categorization on reversed lens images. Setups folder, gets images, and then callses segmentReverseImage function.
"""

# Base packages 
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os


# Import OpenCV
import cv2

# Import functions from segment_micro.py
import segment_reversed

# Configuration
target_dim = 52
baf3_categorization = False
source_directory = "./data/20170425/reversed_lens/10um/"
output_directory = "./data/20170425/reversed_lens/10um/segmented/"

# Key_Press : Folder Name of class
classes = {
	"1": "10um-particle", 
	"2": "10um-clump",
	"0": "10um-other", 
	"5": "10um-discard"
}

# Make Output Folder
os.mkdir(output_directory)

# Create class folders in output_directory
for _, class_name in classes.iteritems():
	os.mkdir(output_directory + class_name)

# Obtain images from source_directory
file_list = glob(source_directory + "*.bmp")


for input_file in file_list: 
	if baf3_categorization:
		segment_reversed.segmentReversed_Baf3(input_file, output_directory, target_dim, classes)
	else:
		segment_reversed.segmentReversed_Micro(input_file, output_directory, target_dim, classes)


