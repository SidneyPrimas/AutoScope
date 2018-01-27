# Import basic libraries
import os
import numpy as np
import glob

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
# Files/Folders
root_folder = "./urine_particles/data/clinical_experiment/prediction_folder/sol2_rev1/"

class_dict =  {0:'10um', 1:'other', 2:'rbc', 3:'wbc'}
other_index = 1
count = 15

# Instantiates configuration for training/validation
config = ClassifyParticles_Config()

# Instantiate training/validation data
data = ClassifyParticlesData(config)

# Create necessary data generators
pred_generator = data.create_prediction_generator(pred_dir_path=root_folder)

# Builds model
model = createModel(input_shape = config.image_shape, base_weights = config.imagenet_weights_file, classes=config.nclasses)

# Load weights (if the load file exists)
CNN_functions.load_model(model, config.weight_file_input, config)

# Predict
all_pred = data.predict_particle_images(model, pred_generator, count=count)

image_predictions = np.argmax(all_pred, axis=1)
prediction_summary = np.zeros((config.nclasses), dtype=float)
for predicted_class in image_predictions: 
	prediction_summary[predicted_class] += 1

# Output results
config.logger.info("Prediction Results")
config.logger.info(image_predictions)
config.logger.info("Total Images: %d", np.sum(prediction_summary))
for key, class_name in class_dict.iteritems():
	config.logger.info("%s : %d", class_name,  prediction_summary[key])

config.logger.info("Particle Counts")
total_particles = np.sum(np.delete(prediction_summary, other_index))
config.logger.info("Total Particles: %d", total_particles)
for key, class_name in class_dict.iteritems():
	if key == other_index:
		continue
	particle_percent = 100*prediction_summary[key]/total_particles
	config.logger.info("%s : %f (%d particles)", class_name, particle_percent, prediction_summary[key])



