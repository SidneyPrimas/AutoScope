import numpy as np
from glob import glob
import cv2
import json

""" 
Implementation Notes: 
+ CV2 loads images as BGR. 
"""


""" User Updated Configuration Parameters"""
input_dir_root = './urine_particles/data/clinical_experiment/raw_image_data/'
input_particle_folders = [
	"wbc/", 
	"10um/", 
	"rbc/", 
]
output_log_path = input_dir_root + "BGR_pixel_means.log"

all_image_paths = []
for target_folder in input_particle_folders:
	all_image_paths.extend(glob(input_dir_root + target_folder + "original/*.bmp"))

all_images = []
for image_path in all_image_paths: 
	img = cv2.imread(image_path)
	all_images.append(img)

# Convert to numpy array for rapid calculations
all_images = np.array(all_images)
print all_images.shape
mean_values = all_images.mean(axis=(0,1,2))

# Mean values is given in Blue, Green, Red order. 
dataset_metadata = {
	"blue": mean_values[0], 
	"green": mean_values[1], 
	"red": mean_values[2]
}
# Create log
print "BGR Means: "
print mean_values
output_log = open(output_log_path , 'w+')
json.dump(dataset_metadata, output_log)
output_log.close()

