"""
    File name: compare_classes.py
    Author: Sidney Primas
    Date created: 11/15/2017
    Python Version: 2.7
    Description: Used to visualize data from experiments (comparing different classes within the same log). 
    To reduce variation in data, average across multiple data points in graph. 
    Errors are probably due to partially complete logs. Edit the log to fix this. 
"""

import matplotlib.pyplot as plt
from scipy import signal
import pandas as PD
import numpy as np
import re
import sys
import confusion_utility

# Import homebrew functions
base_directory = "./classify_particles_tf/data/log/20171026_CICS_resampling/size_52px/"
log_name = "CICS_resample_base_1p36um_52px_v2"
sys.path.append(base_directory)
import config 

# Window size for smoothing data
window_input = 20
title = ""


log_path = base_directory + log_name
total_images = []
class_accuracy = []



### Extract relevant data from the logs ###
with open(log_path) as f:
	file_lines = f.readlines()
	max_lines = len(file_lines)
	for i, line in enumerate(file_lines):
		# Break if the entire confusion matrix is not present in log. 
		if (i+config.logs['num_of_classes']+2 > max_lines): break

		# Process summary line
		if (line.find("Step: ") >= 0):
			total_images.extend(re.findall("Images Trained: (\d+)", line))

		# Process confusion matrix
		if (line.find("[[") >= 0):
			# Convert confusion matrix from a string to np_array (using homebrew utility functions)
			cnf_matrix = confusion_utility.convert_to_np_array(file_lines[i:i+config.logs['num_of_classes']])
			# Calculate accuracy
			cnf_accuracy = np.diag(100*cnf_matrix.astype('float') / cnf_matrix.sum(axis=1))
			class_accuracy.append(cnf_accuracy)



### Convert from Strings to Numbers (float or int), and append to global struct
total_images = map(int, total_images)

### Extract each class accuracy into it's own list. 
# hsplit returns a list of np.arrays, where each np.array is a column of the original
class_accuracy_split =  np.hsplit(np.array(class_accuracy), config.logs['num_of_classes'])

### Moving Average Filters: Store data in class_accuracy_smooth. 
# Only use pandas for simple moving averaging functino. Convert back to numpy array after for consistency. 
# Since we are smoothing, the first window_input number of values in the arrays will be NaN
class_accuracy_smooth = []
for class_array in class_accuracy_split: 
	class_pd = PD.Series(map(float, class_array))
	class_smooth = class_pd.rolling(window=window_input,center=False).mean()
	class_accuracy_smooth.append(PD.Series.as_matrix(class_smooth))


# PLOT THE GRAPHS
for n in range(len(class_accuracy_smooth)):
	# Print accuracy stats. 
	print " Res of %s: Number of Images (%d), Accuracy (%f)"%(config.logs['class_names'][n], total_images[-1], class_accuracy_smooth[n][-1])

	# Visualize the accuracy (across all data)
	fig = plt.figure(1)
	plt.plot(total_images, class_accuracy_smooth[n], label = config.logs['class_names'][n], linewidth=2.0)
	fig.patch.set_facecolor('white')
	plt.title(title, fontsize="20")
	plt.xlabel("Number of Training Cycles", fontsize="20")
	plt.ylabel("Classification Accuracy (%)", fontsize="20")
	plt.legend(loc='center right', prop={'size':12}, frameon=False)
	x1,x2,y1,_ = plt.axis()
	plt.axis((x1,x2,0,100))

plt.show()