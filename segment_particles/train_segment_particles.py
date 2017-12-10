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
from model_FCN8 import FCN8_32px_factor as createModel
from SegmentParticles_config import SegmentParticles_Base



# Instantiates configuration for training/validation
config = SegmentParticles_Base()

# Instantiate training/validation data
data = SegmentParticlesData(config)

# Print configuration
CNN_functions.print_configurations(config) # Print config summary to log file
data.print_data_summary() # Print data summary to log file


# Builds model
model = createModel(input_shape = config.image_shape, base_weights = config.imagenet_weights_file, classes=config.nclasses)

# Load weights (if the load file exists)
CNN_functions.load_model(model, config.weight_file_input, config)

# Train
data.train_entire_model(model)


