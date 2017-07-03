"""
    File name: micro_categorization.py
    Author: Sidney Primas
    Date created: 05/03/2017
    Python Version: 2.7
    Description: Perform segmentation and categorization on microscope images. 
"""

# Base packages 
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os


# Import OpenCV
import cv2

# Import functions from segment_micro.py
import segment_micro

# Configuration
target_dim = 52
source_directory = "./data/20170425/microscope/baf3/baf3_final/"
output_directory = "./data/20170425/microscope/baf3/baf3_final/segmented1/"

# Key_Press : Folder Name of class
classes = {
	"1": "baf3-particle", 
	"2": "baf3-clump",
	"0": "baf3-other"
}

# Make Output Folder
os.mkdir(output_directory)

# Create class folders in output_directory
for _, class_name in classes.iteritems():
	os.mkdir(output_directory + class_name)

# Obtain images from source_directory
file_list = glob(source_directory + "*.tif")


for input_file in file_list: 
	segment_micro.segmentMicroImage(input_file, output_directory, target_dim, classes)