"""
    File name: compare_logs.py
    Author: Sidney Primas
    Date created: 11/15/2017
    Python Version: 2.7
    Description: Compare multiple logs. 
"""

import matplotlib.pyplot as plt
from scipy import signal
import pandas as PD
import numpy as np
import re
import sys

# Import homebrew functions
sys.path.append("./segment_particles/data/CICS_experiment/log/end_to_end_test/")
import config 

### Execution Notes ####
# Be careful, truncation algo truncates data based on index, and not based on images processes (as it should)
# ToDo: Change truncation algo to truncate by images processes instead of results printed in log. 



min_image_index = 0 # Use to track the log with the least images. 
# Variables that aggregate data across all logs. 
all_total_images = []
all_accuracy = []
all_batch_loss = []

### Extract relevant data from the logs ###
for file_name in config.logs['file_name']: 

	filename = config.log_directory + file_name

	# Variables that aggregate data from single log. 
	total_images = []
	accuracy = []
	batch_loss = []

	# Parse through log to pull out summary results. 
	with open(filename) as f:
		for line in f:
			# Process summary line
			if (line.find("Step: ") >= 0):
				total_images.extend(re.findall("Images Trained: (\d+)", line))
				accuracy.extend(re.findall("Training accuracy: (\d\.\d+)", line))
				batch_loss.extend(re.findall("Batch loss: (\d+\.\d+)", line))


	# Track the number of iterations (in terms of images) that all datasets have gotten to. 
	# Allows for comparison across datasets based on the number of images used for training. 
	if ((min_image_index == 0) or (min_image_index > len(total_images))):
		min_image_index = len(total_images)

	# Convert from Strings to Numbers (float or int). Then, append to global struct.
	total_images = map(int, total_images)
	all_total_images.append(total_images)
	accuracy = PD.Series(map(float, accuracy))
	accuracy = accuracy.rolling(window=3,center=False).mean() # Rolling Average (use built in pandas function)
	all_accuracy.append(accuracy)
	batch_loss = map(float, batch_loss)
	all_batch_loss.append(batch_loss)




# PLOT THE GRAPHS
for n in range(len(all_total_images)):

	# Print accuracy stats. 
	print " Res of %s: Number of Images (%d), Accuracy (%f)"%(config.logs['file_name'][n], all_total_images[n][min_image_index-1],all_accuracy[n][min_image_index-1])
	print "Total Images (%d), Final Accuracy (%f)"%(all_total_images[n][-1], all_accuracy[n].iloc[-1])
	print "Final Accuracy (%f)"%(all_accuracy[n].iloc[-1])

	# Visualize the accuracy (across all data)
	fig = plt.figure(1)
	plt.plot(all_total_images[n], all_accuracy[n]*100, label = config.logs['legend'][n], linewidth=2.0)
	fig.patch.set_facecolor('white')
	plt.xlabel("Number of Training Cycles", fontsize="20")
	plt.ylabel("Classification Accuracy (%)", fontsize="20")
	plt.legend(loc='center right', prop={'size':12}, frameon=False)
	x1,x2,y1,_ = plt.axis()
	plt.axis((x1,x2,0,100))

	# Visualize the accuracy (across all data)
	fig = plt.figure(2)
	plt.plot(all_total_images[n][:min_image_index], all_accuracy[n][:min_image_index]*100, label = config.logs['legend'][n], linewidth=2.0)
	fig.patch.set_facecolor('white')
	plt.xlabel("Number of Training Cycles", fontsize="20")
	plt.ylabel("Classification Accuracy (%)", fontsize="20")
	plt.legend(loc='center right', prop={'size':12}, frameon=False)
	x1,x2,y1,_ = plt.axis()
	plt.axis((x1,x2,0,100))


plt.show()