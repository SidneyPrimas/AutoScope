"""
    File name: resampling_visuals.py
    Author: Sidney Primas
    Date created: 4/17/2017
    Python Version: 2.7
    Description: Used for the Resolution Threshold Experiment. Select the logs that you want to graph by uncommenting the correct section.
    To reduce variation in data, average across multiple data points in graph. 
    Errors are probably due to partially complete logs. Edit the log to fix this. 
"""

import re
import matplotlib.pyplot as plt
from scipy import signal
import pandas as PD

## Different Classes
# Storage structure: [EC2_address, destination_file_name, resolution]

# Servers that contain particle images with *VARIABLE* sizes at 4 classes. 
# aws_servers = [
# 	["ubuntu@ec2-52-14-207-70.us-east-2.compute.amazonaws.com", 	"log_0p54", 		0.54],
# 	["ubuntu@ec2-52-14-255-209.us-east-2.compute.amazonaws.com", 	"log_0p7", 			0.7],
# 	["ubuntu@ec2-52-14-249-89.us-east-2.compute.amazonaws.com", 	"log_1p2", 			1.2],
# 	["ubuntu@ec2-52-15-187-153.us-east-2.compute.amazonaws.com", 	"log_1p4", 			1.4], 
# 	["ubuntu@ec2-13-58-42-204.us-east-2.compute.amazonaws.com",		"log_1p7",			1.7],
# 	["ubuntu@ec2-52-15-116-80.us-east-2.compute.amazonaws.com", 	"log_2p0", 			2.0], 
# 	["ubuntu@ec2-13-58-115-146.us-east-2.compute.amazonaws.com", 	"log_2p5", 			2.5], 
# 	["ubuntu@ec2-52-14-62-195.us-east-2.compute.amazonaws.com", 	"log_4p0", 			4]
# ]

#Servers that contain particle images with *FIXED* sizes at 4 classes. 
aws_servers = [
	#["ubuntu@ec2-52-15-156-226.us-east-2.compute.amazonaws.com", 	"log_52px_0p54", 	0.54], 
	["ubuntu@ec2-52-14-216-165.us-east-2.compute.amazonaws.com", 	"log_52x_0p7",		"0.7 um/px"],
	["ubuntu@ec2-13-58-5-196.us-east-2.compute.amazonaws.com",		"log_52px_1p4", 	"1.4 um/px"], 
	["ubuntu@ec2-13-58-130-173.us-east-2.compute.amazonaws.com", 	"log_52px_2p5", 	"2.5 um/px"], 
	["ubuntu@ec2-52-15-183-108.us-east-2.compute.amazonaws.com", 	"log_52px_4p0", 	"4.0 um/px"],
	["ubuntu@ec2-52-14-255-109.us-east-2.compute.amazonaws.com", 	"log_52px_8p0", 	"8.0 um/px"],
	#["ubuntu@ec2-13-58-9-117.us-east-2.compute.amazonaws.com", 		"log_52px_10p0", 	"10.0 um/px"], 
	["ubuntu@ec2-13-58-78-86.us-east-2.compute.amazonaws.com", 		"log_52px_12p0", 	"12.0 um/px"],  
	#["ubuntu@ec2-13-58-74-4.us-east-2.compute.amazonaws.com", 		"log_52px_16p0", 	"16.0 um/px"]		
]

#Servers that contain particle images with *FIXED* sizes at 5 classes. 
# aws_servers = [
# 	["ubuntu@ec2-52-14-144-27.us-east-2.compute.amazonaws.com", 	"log_52px_0p54_5C", 	0.54], 
# 	["ubuntu@ec2-13-58-11-157.us-east-2.compute.amazonaws.com", 	"log_52px_4p0_5C", 		4.0]		
# ]

log_directory =  "./log/downsampling_experiment/"

title = "Accuracy vs. Training Cycles for Different Image Resolutions"



min_image_index = 0
all_total_images = []
all_accuracy = []
all_batch_loss = []

### Extract relevant data from the logs ###
for server in aws_servers: 

	filename = log_directory + server[1]

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
	accuracy = accuracy.rolling(window=10,center=False).mean()
	all_accuracy.append(accuracy)
	batch_loss = map(float, batch_loss)
	all_batch_loss.append(batch_loss)




# PLOT THE GRAPHS
for n in range(len(all_total_images)):

	#out = 438 if 438 < len(all_total_images[n]) else len(all_total_images[n])
	print " Res of %s: Number of Images (%d), Accuracy (%f)"%(aws_servers[n][1], all_total_images[n][min_image_index-1],all_accuracy[n][min_image_index-1])
	print "Total Images (%d), Final Accuracy (%f)"%(all_total_images[n][-1], all_accuracy[n].iloc[-1])
	# Visualize the accuracy
	plt.figure(1)
	plt.plot(all_total_images[n], all_accuracy[n], label = aws_servers[n][2])
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