"""
    File name: indicate_coordinates_sandbox.py
    Author: Sidney Primas
    Date created: 06/25/2017
    Python Version: 2.7
    Description: Indicate selected coordinates (from input file) on image
"""


# Base packages 
import numpy as np
import matplotlib.pyplot as plt
import json

# Import OpenCV
import cv2

# Configurable Variables
input_folder = "./data/20170425/reversed_lens/6um/"
input_file = "1.bmp"
target_dim =  52

# Load image
im_original = cv2.imread(input_folder+input_file, cv2.IMREAD_GRAYSCALE)
im_orignal_max_px = int(np.amax(im_original))

# Open file for reading
log = open("./data/log/log_2017-06-25_17:24:57", 'r')
# Load the data structure saved in JSON format. 
particle_list = json.load(log)
log.close()

# Indicate selected particles 
for centroid in particle_list:

	# Draw a rectangle around each segmented particle. 
	# Find the corners (opposite sides) of the rectangle. They need to be integer tuples. 
	x1 = int(centroid[0]-target_dim/2)
	y1 = int(centroid[1]-target_dim/2)
	x2 = int(centroid[0]+target_dim/2)
	y2 = int(centroid[1]+target_dim/2)


	# Applying cropping rectangles to any particle that we detected. 
	#Python: cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> None
	cv2.rectangle(im_original, (x1,y1), (x2,y2), color=im_orignal_max_px, thickness=4)


# Show and save input image with selected particles shown
fig = plt.figure()
plt.title("Final Output")
imgplot = plt.imshow(im_original, cmap='gray', interpolation='nearest')
plt.axis('off')

plt.show()

#output_image_name = "selected_%s"%(input_file)
#cv2.imwrite(input_folder+output_image_name, im_original, [cv2.IMWRITE_JPEG_QUALITY, 100])