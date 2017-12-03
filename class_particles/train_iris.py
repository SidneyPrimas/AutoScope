# Import basic libraries
import os
import numpy as np
import glob

#Import keras libraries
from tensorflow.python.keras.optimizers import SGD, RMSprop
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.utils import plot_model

# import from local libraries
from IrisData import IrisData
import CNN_functions
#from model_factory_particles import VGG16_with_custom_FC_average as createModel
#from model_factory_particles import VGG16_with_custom_FC_flatten as createModel
#from model_factory_particles import VGG16_bottom3_layers_custom_FC_average as createModel
#from model_factory_particles import VGG16_bottom3_layers_custom_FC_flatten as createModel
from model_factory_particles import VGG16_bottom2_layers_custom_FC_average as createModel
from Iris_config import IrisConfig_Base



# Instantiates configuration for training/validation
config = IrisConfig_Base()

# Instantiate training/validation data
data = IrisData(config)

# Print configuration
CNN_functions.print_configurations(config) # Print config summary to log file
data.print_data_summary() # Print data summary to log file


# Builds model
model = createModel(input_shape = config.image_shape, base_weights = config.imagenet_weights_file, classes=config.classes)

# Load weights (if the load file exists)
CNN_functions.load_model(model, config.custom_weight_file_input, config)


#data.transfer_learn_train(model)
#data.fine_tune_train(model)
data.train_entire_model(model)




