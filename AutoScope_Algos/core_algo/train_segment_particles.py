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


# Instantiates configuration for training/validation
config = SegmentParticles_Config()

# Configuration sanity check
CNN_functions.validate_segmentation_config(config)

# Instantiate training/validation data
data = SegmentParticlesData(config)

# Create necessary data generators
train_generator = data.get_data_generator(config.train_images_dir, config.train_annotations_dir) 
val_generator = data.get_data_generator(config.val_images_dir, config.val_annotations_dir)

# Print configuration
CNN_functions.print_configurations(config) # Print config summary to log file
data.print_data_summary() # Print data summary to log file


# Builds model
model = createModel(input_shape = config.image_shape, base_weights = config.imagenet_weights_file, classes=config.nclasses)

# Load weights (if the load file exists)
CNN_functions.load_model(model, config.weight_file_input, config)

# Train
data.train_entire_model(
	model = model, 
	train_generator = train_generator, 
	val_generator = val_generator)

