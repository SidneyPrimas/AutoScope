import tensorflow as tf
import numpy as np

# Homebrew libraries
from ParticleDataset import ParticleDataset
from session_config import SessionConfiguration_Iris
from model import ParticleDetectionModel
import utility_functions

"""
Description: 
Execution: 
ToDo:
+ Count in TF: Count number of training/validation iterations within tensorflow. Allows to easily start/stop training. 
+ Start/Stop Training: Create infrastructure to start/stop training seemessly, including from perspective of log file. 
+ Sparse Softmax (possible): Switch to sparse softmax for faster computing. 
"""


# Instantiates configuration for training/validation
params = SessionConfiguration_Iris()

# Create data object for getting training/validation data (for Iris and homebrew urine data)
particle_data = ParticleDataset(
	root_dir=params.data_directory, 
	directroy_map=params.directory_map, 
	class_size=params.class_size, 
	target_dim=params.target_dim, 
	resize=params.resize_images
	)

# Define weights (set loss weighting based on proportion of images in each class).
# params.weights = utility_functions.get_loss_weights(particle_data, params)

# Print log header
utility_functions.print_log_header(particle_data, params)


# Create graph context (new, clean graph)
with tf.Graph().as_default():

	# Build model
	model = ParticleDetectionModel(params)

	# Op to create tensors created in ParticleDetectionModel
	init_op = tf.global_variables_initializer()

	# Create session to run graph. 
	with tf.Session() as session:
		
		session.run(init_op) #initialize tf variables		
		model.restore_graph_variables(session) # The initialized vars are overwritten from checkpoint file. 
		model.train_forever(session, particle_data)


	

