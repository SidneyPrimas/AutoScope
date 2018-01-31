# Import basic libraries
import os
import numpy as np
import glob
import math

#Import keras libraries
from tensorflow.python.keras.optimizers import SGD, RMSprop
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.utils import plot_model

# import from local libraries
from ClassifyParticlesData import ClassifyParticlesData
import CNN_functions
from classification_models import base_model as createModel
from ClassifyParticles_config import ClassifyParticles_Config

"""
Description: 

Execution Notes: 


To Do: 
"""

""" Configuration """
# User Updated
root_folder = "./urine_particles/data/clinical_experiment/prediction_folder/sol1_rev1/"
input_img_count = 6

# Files/Folders
segmentation_folder = "cropped_output/"
image_data_folder = root_folder + segmentation_folder + "data/"
sorted_output = root_folder + segmentation_folder + "sorted_output/"

class_mapping =  {0:'10um', 1:'other', 2:'rbc', 3:'wbc'}
throw_out_class_index = 1

# Microscope configuration
HPF_area = 0.196349541 # in mm^2
primas_area = 10.1568 # in mm^2
conversion_factor = HPF_area/primas_area # converts from Primas unit to HPF unit. 


# Instantiates configuration for training/validation
config = ClassifyParticles_Config()

# Instantiate training/validation data
data = ClassifyParticlesData(config)

# Create necessary data generators
pred_generator = data.create_custom_prediction_generator(pred_dir_path=image_data_folder)

# Builds model
model = createModel(input_shape = config.image_shape, base_weights = config.imagenet_weights_file, classes=config.nclasses)

# Load weights (if the load file exists)
CNN_functions.load_model(model, config.weight_file_input, config)

# Predict
total_cropped_images = len(glob.glob(image_data_folder + "images/*.bmp"))
num_batches = int(math.floor(total_cropped_images/float(data.config.batch_size)))
all_pred = data.predict_particle_images(
	model=model, 
	pred_generator=pred_generator, 
	total_batches=num_batches, # Due to keras convention, if the images cannot be evenly split into batches, the last batch will only be partially filled.
	output_dir=sorted_output, 
	labels_to_class=class_mapping) 

image_predictions = np.argmax(all_pred, axis=1)
prediction_summary = np.zeros((config.nclasses), dtype=float)
for predicted_class in image_predictions: 
	prediction_summary[predicted_class] += 1



# Output results
config.logger.info("Prediction Results")
config.logger.info("Total Cropped Images: %d", np.sum(prediction_summary))
total_particles = np.sum(np.delete(prediction_summary, throw_out_class_index))
config.logger.info("Total Particles (w/o other): %d", total_particles)
config.logger.info("\nType\t\t\tCount\t\t\tperPrimas\t\t\tperHPF\t\t\tPercent of Particles")
for key, class_name in class_mapping.iteritems():
	perPrimas = prediction_summary[key]/float(input_img_count) #calculates average per Primas image
	perHPF = perPrimas*conversion_factor # converts to perHPF
	particle_percent = 100*prediction_summary[key]/total_particles
	particle_percent_str = '%.02f%%'%(particle_percent)
	if key == throw_out_class_index:
		particle_percent_str = "N/A"
	config.logger.info("%s\t\t\t%d\t\t\t%.02f\t\t\t\t%.02f\t\t\t%s", class_name, prediction_summary[key], perPrimas, perHPF, particle_percent_str)







