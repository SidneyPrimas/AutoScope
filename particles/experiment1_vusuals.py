"""
    File name: experiment1_visuals.py
    Author: Sidney Primas
    Date created: 5/10/2017
    Python Version: 2.7
    Description: Used to visualize data from experiments. 
    To reduce variation in data, average across multiple data points in graph. 
    Errors are probably due to partially complete logs. Edit the log to fix this. 
"""

import re
import matplotlib.pyplot as plt
from scipy import signal
import pandas as PD

## Different Classes
# Storage structure: [EC2_address, destination_file_name, resolution]

files = [ 
		"microscope_output/particles_clumps_other2", 
		"reversed_output/particles_clumps_other",
		"reversed_output/switched_6um_and_10um", 
		"reversed_output/random_validation_files"
		]


legend = [
		"Images from Microscope", 
		"Images from Reversed Lens", 
		"Reversed Lens where Switched 6um and 10um in Validation Dataset",
		"The Validation Images Are Randomized"
]

log_directory =  "./log/"

title = "Accuracy vs. Training Cycles"



min_image_index = 0
all_total_images = []
all_accuracy = []
all_batch_loss = []

### Extract relevant data from the logs ###
for file_name in files: 

	filename = log_directory + file_name

	total_images = []
	accuracy = []
	batch_loss = []

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

	# Convert from Strings to Numbers (float or int), and append to global struct
	total_images = map(int, total_images)
	all_total_images.append(total_images)
	## Median filter or Moving average filter
	#accuracy = signal.medfilt(map(float, accuracy), 3)
	accuracy = PD.Series(map(float, accuracy))
	accuracy = accuracy.rolling(window=7,center=False).mean()
	all_accuracy.append(accuracy)
	batch_loss = map(float, batch_loss)
	all_batch_loss.append(batch_loss)




# PLOT THE GRAPHS
for n in range(len(all_total_images)):

	#out = 438 if 438 < len(all_total_images[n]) else len(all_total_images[n])
	print " Res of %s: Number of Images (%d), Accuracy (%f)"%(files[n], all_total_images[n][min_image_index-1],all_accuracy[n][min_image_index-1])
	print "Total Images (%d), Final Accuracy (%f)"%(all_total_images[n][-1], all_accuracy[n].iloc[-1])
	# Visualize the accuracy
	plt.figure(1)
	plt.plot(all_total_images[n], all_accuracy[n], label = legend[n])
	plt.title(title)
	plt.xlabel("Training Iterations (in number of images)")
	plt.ylabel("Accuracy")
	plt.legend(loc='lower right', prop={'size':12})

	# Visualize the batch loss
	# plt.figure(2)
	# plt.plot(all_total_images[n], all_batch_loss[n], label=aws_servers[n][1])
	# plt.title("Batch Loss vs. Total Images Trained")
	# plt.xlabel("Total Images Used for Training")
	# plt.ylabel("Batch Loss")
plt.show()