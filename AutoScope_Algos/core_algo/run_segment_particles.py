# Import basic libraries
import os
import numpy as np
import glob

#Import keras libraries
from tensorflow.python.keras.optimizers import SGD, RMSprop
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.utils import plot_model

# import from local libraries
from SegmentParticlesData import SegmentParticlesData
import CNN_functions
from segmentation_models import FCN8_32px_factor as createModel
from SegmentParticles_config import SegmentParticles_Config

# User inputs (apply to any directories)
custom_val_images_path = "./urine_particles/data/clinical_experiment/image_data/20180120_training/segmentation/val_images/"
custom_val_annotations_path = "./urine_particles/data/clinical_experiment/image_data/20180120_training/segmentation/val_annotations/"


# Instantiates configuration for training/validation
config = SegmentParticles_Config()

# Configuration sanity check
CNN_functions.validate_segmentation_config(config)

# Instantiate training/validation data
data = SegmentParticlesData(config)

# Create necessary data generators
val_generator = data.get_data_generator(custom_val_images_path, custom_val_annotations_path)


# Print configuration
CNN_functions.print_configurations(config) # Print config summary to log file
data.print_data_summary() # Print data summary to log file


# Builds model
model = createModel(input_shape = config.image_shape, base_weights = config.imagenet_weights_file, classes=config.nclasses)

# Load weights (if the load file exists)
CNN_functions.load_model(model, config.weight_file_input, config)

# Validate or Predict
data.validate_epoch(
	model = model, 
	val_generator = val_generator,
	get_particle_accuracy = True)


