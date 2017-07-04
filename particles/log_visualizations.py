import re
import matplotlib.pyplot as plt
from scipy import signal
import pandas as PD

## Different Classes
# files = [		"./log/saved_logs/log_4classesOther_basicGraph_imagesPerClass_newImages_111", 
# 				"./log/saved_logs/log_6classesOther_basicGraph_imagesPerClass_newImages_163", 
# 				"./log/saved_logs/log_7classes_other_basicGraph_imagesPerClass_newImages_179"]
# legend_names = ["4 Classes", 
# 				"6 Classes", 
# 				"7 Classes"]

# title = "Accuracy vs. Training Iterations for Different Class Sizes"

## Different Architectures
files = [	"./log/saved_logs/log_6classesOther_basicGraph_imagesPerClass_newImages_163",
			"./log/saved_logs/log_6lcassesOther_2by2C-Layers_imagesPerClass_newImages_122"]
legend_names = ["Single 5x5px Kernel",
				"Two 3x3px Kernels"]

title = "Accuracy vs. Training Iterations \nfor Different Convolutional Layer Implementations"

## Augmented Data
# files = ["./log/saved_logs/log_4classes_basicGraph_imagesPerClass_augmented_193",
# 		"./log/saved_logs/log_6classes_basicGraph_imagesPerClass_augmented_178",
# 		"./log/saved_logs/log_7classes_basicGraph_imagesPerClass_augmented_185",
# 		"./log/saved_logs/log_6lcasses_2by2C-Layers_imagesPerClass_augmented_154"]
		

# legend_names = ["4 Classes (Base Graph) ",
# 				"6 Classes (Base Graph)", 
# 				"7 Classes (Base Graph)",
# 				"6 Classes (Two 3x3 Kernels)"]

# title = "Accuracy vs Training Iterations for 32x Augmented Dataset"

# Training variations
# files = ["./log/saved_logs/log_6classesOther_basicGraph_imagesPerClass_newImages_163",
# 		"./log/saved_logs/log_6classesOther_basicGraph_Random_newImages_withWeights_178",
# 		"./log/saved_logs/log_6classesOther_basicGraph_Random_newImages_noWeights_174"]

# legend_names = ["Equal Images Per Class",
# 				"Randomly Selected (weighted)",
# 				"Randomly Selected (not weighted)"]

# title = "Accuracy vs. Training Iterations for \n Different Image-Feeding Approaches (6 Classes)"




min_image_index = 0
all_total_images = []
all_accuracy = []
all_batch_loss = []

for filename in files: 


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


	# Track the larget image
	if ((min_image_index == 0) or (min_image_index > len(total_images))):
		min_image_index = len(total_images)

	# Convert from Strings to Numbers (float or int), and append to global struct
	total_images = map(int, total_images)
	all_total_images.append(total_images)
	## Median filter or Moving average filter
	#accuracy = signal.medfilt(map(float, accuracy), 3)
	accuracy = PD.Series(map(float, accuracy))
	accuracy = accuracy.rolling(window=2,center=False).mean()
	all_accuracy.append(accuracy)
	batch_loss = map(float, batch_loss)
	all_batch_loss.append(batch_loss)




# PLOT THE GRAPHS
for n in range(len(all_total_images)):

	#out = 438 if 438 < len(all_total_images[n]) else len(all_total_images[n])
	print "%s: Number of Images (%d), Accuracy (%f)"%(legend_names[n], all_total_images[n][min_image_index-1],all_accuracy[n][min_image_index-1])
	print "Total Images (%d), Final Accuracy (%f)"%(all_total_images[n][-1], all_accuracy[n].iloc[-1])
	# Visualize
	plt.figure(1)
	plt.plot(all_total_images[n], all_accuracy[n], label = legend_names[n])
	plt.title(title)
	plt.xlabel("Training Iterations (in number of images)")
	plt.ylabel("Accuracy")
	plt.legend(loc='center right', prop={'size':12})


	# plt.figure(2)
	# plt.plot(all_total_images[n], all_batch_loss[n], label=legend_names[n])
	# plt.title("Batch Loss vs. Total Images Trained")
	# plt.xlabel("Total Images Used for Training")
	# plt.ylabel("Batch Loss")
plt.show()