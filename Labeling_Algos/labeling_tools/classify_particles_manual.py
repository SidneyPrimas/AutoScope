# Base packages 
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
from glob import glob

# Configurations
target_log = "img5_sol3_rbcTrain_coordinates.json"
main_class = "rbc"
root_dir = "./data/20180120/rbc_half_rev1/selected/"



coordinates_folder = root_dir + "coordinates/"
input_image_dir = root_dir + "original/"
target_dim = 64
# Key_Press : meta-data of class
# Notes: num parameter needs to be in numerical order, starting from 1. 0 is the background. 
classes = { 
	"1": {"main":main_class, "sub":"particle",	"num":1}, 
	"0": {"main":main_class, "sub":"other",		"num":2},
	"s": {"main":main_class, "sub":"discard", 	"num":3},
	"a": {"main":main_class, "sub":"accident", 	"num":4}
}

def main(): 
	log_path = coordinates_folder + target_log
	classify_particle(log_path)


def classify_particle(log_path):

	# Define directory/file names
	image_name = get_image_file_name(log_path)
	inputImage_path = input_image_dir + image_name + ".bmp"

	# Load image
	im_original = cv2.imread(inputImage_path) # Careful: format in BGR


	# Load coordinates log
	input_log = open(log_path , 'r')
	particle_list = json.load(input_log)
	particle_class_list = []
	input_log.close()

	# Manually classify particles. 
	for index in range(len(particle_list)):

		# Find the corners (opposite sides) of the rectangle. They need to be integer tuples. 
		centroid = particle_list[index]
		x1 = int(centroid[0]-target_dim/2)
		y1 = int(centroid[1]-target_dim/2)
		x2 = int(centroid[0]+target_dim/2)
		y2 = int(centroid[1]+target_dim/2)


		# Applying cropping rectangles to any particle that we detected. 
		im_output = im_original.copy() # So only single boxed particle is shown to consumer. 
		cv2.rectangle(im_output, (x1,y1), (x2,y2), color=(0,0,255), thickness=2)


		# Manual cetegorization
		fig = plt.figure()
		im_output_plt = cv2.cvtColor(im_output, cv2.COLOR_BGR2RGB)
		imgplot = plt.imshow(im_output_plt, interpolation='nearest')
		plt.axis('off')
		title_text = [[key, particle_info["sub"]]for key, particle_info in classes.iteritems()]
		plt.title(title_text, fontsize = 8)
		# Zoom into correct area of image
		plt.axis([x1-target_dim*4,x2+target_dim*4,y2+target_dim*4, y1-target_dim*4]) # y-axis is flipped. 


		# Categorize the image based on user keypress
		plt.show(block=False)
		keypress = raw_input(">")

		if keypress == "s": # skip particle
			particle_class = classes["s"]
			particle_class_list.append([particle_list[index], particle_class])
			plt.close(fig)
			continue
		elif keypress == "break": # break out of image. Go to next image. 
			plt.close(fig)
			break
		elif keypress in classes:
			particle_class = classes[keypress]
			particle_class_list.append([particle_list[index], particle_class])
		# If the keypress isn't valid, then place the picture in accident class. 
		else:
  			particle_class = classes["a"]
			particle_class_list.append([particle_list[index], particle_class])

		plt.close(fig)

	# Dumpe data to json
	json_data = {
		"classes": classes, 
		"particle_list": particle_class_list
	}
	output_log_path = coordinates_folder  + image_name + "_classes.json"
	output_log = open(output_log_path , 'w+')
	json.dump(json_data, output_log)
	output_log.close()


def get_image_file_name(log_path):
	fileName_start = log_path.rfind("/")
	fileName_end = log_path.rfind("_coordinates.json")
	image_file_name = log_path[fileName_start+1:fileName_end]
	return image_file_name



if __name__ == "__main__":
    main()