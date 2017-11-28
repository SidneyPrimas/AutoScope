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
from model_factory import VGG16_with_custom_FC1
from Iris_config import IrisConfig_Base

# Instantiates configuration for training/validation
config = IrisConfig_Base()

# Instantiate training/validation data
data = IrisData(config)

# Print configuration
CNN_functions.print_configurations(config) # Print config summary to log file
data.print_data_summary() # Print data summary to log file


# Builds model
model = VGG16_with_custom_FC1(input_shape = config.image_shape, base_weights = config.input_weights_file, classes=config.classes)

# Freeze Lower Layers
CNN_functions.freeze_lower_layers(model, config.tl_freeze_layer)

# Compiles model
# Note on loss: Loss value minimized by individual model is the sum of each model. 
model.compile(optimizer=config.optimizer, loss='categorical_crossentropy', metrics=['accuracy', categorical_accuracy])
model.summary(print_fn = config.logger.info)


# Create data generators
train_generator = data.create_training_generator(save_to_dir_bool = config.save_to_dir_bool)
validation_generator = data.create_validation_generator()

# Trains model
history_tl = model.fit_generator(
	train_generator, # Calling the function once produced a tuple with (inputs, targets, weights). Produces training data for a single batch. 
	epochs=config.epoch,
	steps_per_epoch=  config.batches_per_epoch_train, # The total batches processes to complete an epoch. steps_per_epoch = total_images/batch_size
	validation_data=validation_generator, # Produces validation data. Validation done after each epoch. 
	validation_steps= config.batches_per_epoch_val
)

# Save trained model
model.save(config.output_model_file)

# Output results
config.logger.info(history_tl.history)
#CNN_functions.plot_training(history_tl)

# TODO: Create better logging function. 
CNN_functions.save_history_to_file(history_tl.history, config.log_dir + "history_output.json")



