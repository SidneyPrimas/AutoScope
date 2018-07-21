"""
    File name: generate_training_snapshots.py
    Author: Sidney Primas
    Python Version: 2.7
    Description: Given multiple models from training, generate the prediction output for each model. 
"""

# Import basic libraries 
import numpy as np
import glob
import cv2

# import from local libraries
from SegmentParticlesData import SegmentParticlesData
from segmentation_models import FCN8_32px_factor as createModel
import CNN_functions
from SegmentParticles_config import SegmentParticles_Config

# Import homebrew functions
model_directory = "./core_algo/data/clinical_experiment/model_storage/20180711_seg_demo/"
input_image_dict = {
	"wbc" : "./core_algo/data/clinical_experiment/image_data/20180711_seg_demo/segmentation/val_images/wbc/wbc_img1.bmp",
	"rbc" : "./core_algo/data/clinical_experiment/image_data/20180711_seg_demo/segmentation/val_images/rbc/rbc_img1.bmp"	
}
output = "./core_algo/data/clinical_experiment/image_data/20180711_seg_demo/segmentation/img_output/training_snapshots/"

# Instantiates configuration for training/validation
config = SegmentParticles_Config()

# Print configuration
CNN_functions.print_configurations(config) # Print config summary to log file

# Configuration sanity check
CNN_functions.validate_segmentation_config(config)

# Instantiate training/validation data
data = SegmentParticlesData(config)

# Builds model
model = createModel(input_shape = config.image_shape, base_weights = config.imagenet_weights_file, classes=config.nclasses)

model_list = glob.glob(model_directory + "*")

for model_path in model_list:

	print "Current Model Path: %s"%( model_path)
	index_end =  model_path.rfind('_')
	index_start = model_path.rfind('_', 0, index_end)
	model_count = model_path[index_start+1:index_end]

	# Load weights (if the load file exists)
	CNN_functions.load_model(model, model_path, config)
	
	for particle_name, input_image_path  in input_image_dict.iteritems():
	
		# Predict segmentation of original_img with segmentation model  
		pred_array = data.predict_image(model, input_image_path)
	
		# Convert to labeled image.
		pred_img = CNN_functions.predArray_to_predMatrix(pred_array, data.config.target_size)

		# Produce rgb image for predicted image from semantic segmentation
		pred_img_rgb = CNN_functions.get_color_image(pred_img, nclasses=data.config.nclasses, colors=[(0, 0, 0), (255,255,255)])

		# save images
		output_path = output + particle_name + "_" + model_count + ".jpg"
		cv2.imwrite(output_path, pred_img_rgb)
		print "Processed Image Output: %s"%(output_path)	
