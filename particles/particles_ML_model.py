# Initial ML Model for Particle Classification

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.image as mpimg
import datetime
import pprint
from collections import namedtuple
import matplotlib.gridspec as gridspec

import particles_batch as batch
import ml_model as ml_model

# Global Variables
DEBUG = False
EQUAL_IMAGES_PER_CLASS = True
LOAD_FILTERS = True
LOAD_FC_LAYERS = False
TRAINING = True
VISUALIZE_FILTERS = False

# Current Setup of Neural Network: 
## The neural net currently has 5 layers, with each convolutional layer having two convolutionals built into it. So, counting these sublayers, we have a 7-layer network.

# To Get in Correct State: 
## Change load file path (hardcoded)
## Update the target_dim based on the size of the input image
## Increase iteration number to display/save
## Turn off/on visualization 

# ToDo: 
## Create wrapper function that just creates the basic neural network (this allows us seperate validation and training)
## Create a count of the number of iterations in TF. 
## Switch to sparse softmax (faster computation)
## Create a global debug statement to enable all debug print statements

directory_map = {
    "IRIS-BACT": 0,         #0
    "IRIS-RBC": 1,          #1
    "IRIS-SPRM": 3,         #3
    "IRIS-WBC": 2,          #2
    "IRIS-CLUMP-WBCC": 2,   #2
    "IRIS-CLUMP-YSTS": 3,   #3
    "IRIS-CRYST-CAOX": 3,   #3
    "IRIS-CRYST-CAPH": 3,   #3
    "IRIS-CRYST-TPO4": 3,   #3
    "IRIS-CRYST-URIC": 3,   #3
    "IRIS-HYAL": 3,         #3
    "IRIS-NHYAL-CELL": 3,   #3
    "IRIS-NHYAL-GRAN": 3,   #3
    "IRIS-NSQEP-REEP": 3,   #3
    "IRIS-NSQEP-TREP": 3,   #3
    "IRIS-SQEP": 3,         #3
}


def main():
	### Setup Configuration Parameters ###
	params = ml_model.ML_Model_Parameters()
	params.directory_map = directory_map
	params.train_batch_size = 100
	params.validation_batch_size = 300

	params.step_display = 1
	params.step_save = 1
	params.step_count = 0
	params.image_trained_count = 0

	params.learning_rate = 1e-4
	params.dropout = 0.5
	params.target_dim = 116
	params.class_size = len(np.unique(params.directory_map.values()))

	params.data_directory = "./data/IrisDB_resampling/"
	params.filter_path = "./data/particle_model_filters_2times2Layers"
	params.fc_layers_path = "./data/particle_model_fc_layers-%d"%(params.class_size)
	

	# Create particle data object for getting training/validation data 
	particle_data = batch.ParticleSet(params.data_directory, params.directory_map, params.class_size, params.target_dim)


	# Print log header
	print_log_header(particle_data, params)

	# Reset everything before rerunning the graph (ensures previous graphs are cleared out)
	tf.reset_default_graph()

	# Initiate Session: Interactive sessions allows for interleaving instructions that make graph and run graph. 
	sess = tf.InteractiveSession()

	# Make placeholder variables: These take the training input images, and the corresponding labels
	x = tf.placeholder(tf.float32, shape=[None, params.target_dim * params.target_dim])
	y_ = tf.placeholder(tf.float32, shape=[None, params.class_size])


	# Reshape data: Reshapes each input x from a 2-D Tensor into a 4D Tensor [batch, image_height, image_width, channel]. 
	# Note: -1 used to indicate that this dimension selected so total size remains constant. In this case, just transfer the None. 
	x_image = tf.reshape(x, [-1, params.target_dim, params.target_dim, 1])

	#### LAYER 1: CONVOLUTIONAL LAYER ####
	# Note: Increase channels from 1 to 32 while reducing pixels by 4. 
	# Define the Filter: The weight variable is a 5x5 kernel (the filter). 
	# The convolutional layer transforms 1 input channel (greyscale) to 32 output channels (32 is arbitrary)
	W_conv1_a = weight_variable([3, 3, 1, 32], name='W_conv1_a')
	# Define the bias for each channel. 
	b_conv1_a = bias_variable([32], name='b_conv1_a')

	# Additional convolusional layer
	W_conv1_b = weight_variable([3, 3, 32, 32], name='W_conv1_b')
	b_conv1_b = bias_variable([32], name='b_conv1_b')

	# Perform Convolution: Convolve x_image with W_conv1 filter, and add the offset. 
	# Then, for each channel of each pixel, take the ReLU. 
	# Size of h_conv1: [batch, 52, 52, 32]
	h_conv1_a = tf.nn.relu(conv2d(x_image, W_conv1_a) + b_conv1_a)
	h_conv1_b = tf.nn.relu(conv2d(h_conv1_a, W_conv1_b) + b_conv1_b)
	# Perfomr Max Pooling: Performs max pooling on the input, striding across the data with 2x2 matrices. 
	# Essentially, we reduce each non-overlapping 2x2 matrix into a single data point. 
	# Output has shape of [batch, 26, 26, 32], where 15 is the batch size.
	h_pool1 = max_pool_2x2(h_conv1_b)


	# Calculate the 2nd layer. Here, we increase the channels from 32 input to 64 output. 
	#### LAYER 2: CONVOLUTIONAL LAYER ####
	# Note: Inceasing channels from 32 to 64 while reducing pixels by 4. 
	W_conv2_a = weight_variable([3, 3, 32, 64], name='W_conv2_a')
	b_conv2_a = bias_variable([64], name='b_conv2_a')

	# Additional convolusional layer
	W_conv2_b = weight_variable([3, 3, 64, 64], name='W_conv2_b')
	b_conv2_b = bias_variable([64], name='b_conv2_b')

	h_conv2_a = tf.nn.relu(conv2d(h_pool1, W_conv2_a) + b_conv2_a)
	h_conv2_b = tf.nn.relu(conv2d(h_conv2_a, W_conv2_b) + b_conv2_b)
	h_pool2 = max_pool_2x2(h_conv2_b)

	#### LAYER 3: DENSELEY CONNECTED LAYER ####
	# Note: A denseley connected layer starts using a regression model to translate data into evidence.  
	# Note: Densely Connected Layer allows processing on entire input. 
	# Notes on input: With 2 max-pooling layers condensing 4-px to 1px, we now have the following shape [batch, 13, 13, 64]
	# First, we linearize all input data, including the 64 channels across each pixel. 
	# Then we create 1024 neurons, which represents the output. 
	W_fc1 = weight_variable([params.target_dim/4 * params.target_dim/4 * 64, 1024], name='W_fc1')
	b_fc1 = bias_variable([1024], name='b_fc1')

	# Reshpae h_pool2 into [batch, linearize (im_pixel*channels)]. 
	# In order to allow h_pool2 to be an input in the correct format, we must reshape it. 
	h_pool2_flat = tf.reshape(h_pool2, [-1, params.target_dim/4 * params.target_dim/4 * 64])

	# Implement first regression model, calculating 1024 buckets of evidence. 
	# High Level note: The convolutional layers perform feature detection. And, the regression model turns the features into evidence. 
	# Note: matmul gives us [batch, 1024]. Or, for each input image, we get 1024 evidnece output buckets. 
	# Note: Feature detection is a learned process. 
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	### LAYER 4: DROPOUT LAYER ###
	# Ensures that these layers are not trained in this implementation. 
	# Note: The placeholder allows us to insert the dropout probability during training, and remove it during testing. 
	keep_prob = tf.placeholder(tf.float32)
	# Note: The dropout function drops nodes in graph. 
	# Note: For nodes it keeps, it scales by 1/keep_prob. And sets other elements to 0. This should keep the expected sum the same. 
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


	### Layer 5: READOUT LAYER (REGRESSION LAYER) ####
	# Note: Perform softmax regression on the class_size classes. 
	# Note: Translation matrix that takes 1024 neuron weights as input, and converts it into class_size evidence outputs for each image. 
	W_fc2 = weight_variable([1024, params.class_size], name='W_fc2')
	b_fc2 = bias_variable([params.class_size], name='b_fc2')
	# Note: Implement the regression model that converts inputs (h_fc1_drop) into probably outcomes (y_conv) using W and b as trained variables. 
	# y_conv is a size of [batch, label #s]. Use for softmax, and the loss function. 
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


	### SETUP LOSS FUNCTION AND TRAINING IN GRAPH ###
	# Defines the loss function: Train all W and b in order to minimize loss across all training inputs. 
	# Note on softmax_cross_entropy_with_logits: applies softmax across unnormalized data, and sums across all classes. 
	# Note on sparse_softmax_cross_entropy_with_logits: Used when label can only be a single result per lable (and not a distribution)
	# Notes on tf.reduce_mean: Computes the mean of the batch (so that we only update the variables once per batch). 
	if EQUAL_IMAGES_PER_CLASS: 
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

	else:
		# Determine loss function with weights according to proportion of images. 
		# Create list of weights for each class based on the proportion of images in that class. 
		# Note: The more images for a class the lower the weight. 
		# A weight is selected so that (probably_of_class_appearing)*(weight_for_class) = 1. 
		# In this way, the loss in each class is re-balanced so that each class will see a similar amount of loss per batch. 
		# This is better because othewise the loss that is used to train the NN will be dominated by feedback from a single class. 
		loss_weight_per_class = getLossWeights(particle_data, params)
		pos_weight = tf.constant(loss_weight_per_class, dtype=tf.float32)

		# The weighted cross entropy allows to adjust the loss through a pos_weight factor. 
		# The positive weight factor allows us to adjust the loss based on the target (or true label).
		cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y_conv, y_, pos_weight))

	# Uses Adam optimizer to update all variables in our system
	# When we run train_step, we updated all W and b. 
	train_step = tf.train.AdamOptimizer(params.learning_rate).minimize(cross_entropy)

	# Operations for accuracy calculation. 
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	### INITIALIZE ALL VARIABLES ###
	# Create saver opject. 
	# Note: Since no variables are explicitly specified, saver stores all trainable variables. 
	saver_filters = tf.train.Saver([W_conv1_a, b_conv1_a, W_conv1_b, b_conv1_b, W_conv2_a, b_conv2_a, W_conv2_b, b_conv2_b])

	saver_fc_layers = tf.train.Saver([W_fc1, b_fc1, W_fc2, b_fc2])

	# Before variables can be used within a session, they must be initialized. 
	# Here, we initialize all variables. 
	sess.run(tf.initialize_all_variables())

	# Restore variables (weights) from previous training sessions. 
	# We over-write the initilized values with weights from previous session
	if (LOAD_FILTERS): 
		saver_filters.restore(sess, "./data/saved_models/particle_model_filters_2times2Layers_from_154_v1")
		print >> params.log,("Restored Filters from: %s")%("./data/saved_models/particle_model_filters_2times2Layers_from_154_v1")

	if (LOAD_FC_LAYERS): 
		saver_fc_layers.restore(sess, params.fc_layers_path)
		print >> params.log,("Restored Fully-Connected Layers from: %s")%(params.fc_layers_path)


	### RUN GRAPH TO TRAIN SYSTEM ###
	while (True):
		params.step_count+=1

		# Train
		#Runs a single train step with a single batch using a keep probability of 0.5
		if (TRAINING):
			params.image_trained_count += params.train_batch_size
			# Obtain training data
			data, labels = particle_data.next_batch(params.train_batch_size, validation=False, per_class_order=EQUAL_IMAGES_PER_CLASS)
			train_step.run(feed_dict={x: data, y_: labels, keep_prob: params.dropout})

		# Save model 
		# Save graph nodes after first run, and every step_display iterations after. 
		if (params.step_count%params.step_save == 0) and TRAINING: 
			# Save variables (save filters and fully connected layers seperately)
			saver_filters.save(sess, params.filter_path)
			saver_fc_layers.save(sess, params.fc_layers_path)
			print >> params.log,("Fitler Models saved in file: %s" % params.filter_path)
			print >> params.log,("Fully Connected Models saved in file: %s" % params.fc_layers_path)


		# Display Results
		# Validate and display validation results after first run, and every step_display iterations after. 
		if (params.step_count%params.step_display == 1) or (not TRAINING):

			# Obtain training data
			data, labels = particle_data.next_batch(params.validation_batch_size, validation=True, per_class_order=EQUAL_IMAGES_PER_CLASS)

			# Evaluates the accuracy and cross_entropy (doesn't drop nodes during evaluation)
			# Obtain all values from graph in single session. 
			train_accuracy, train_loss, y_truth, y_pred = sess.run([accuracy, cross_entropy, y_, y_conv],feed_dict={x: data, y_: labels, keep_prob: 1.0})
			
			print >> params.log,("Step: %d, Images Trained: %d, Batch loss: %2.6f, Training accuracy: %1.3f "%(params.step_count, params.image_trained_count, train_loss, train_accuracy))
			
			# Create and confusion matrix
			log_confusion_matrix(y_truth, y_pred, params)


			if (VISUALIZE_FILTERS):
				filters1, filters2 = sess.run([W_conv1_b, W_conv2_b],feed_dict={x: data, y_: labels, keep_prob: 1.0})
				visualizeFilters(filters1)
				visualizeFilters(filters2)

			# Print images used in this batch
			if (DEBUG):
				x_image_out = x_image.eval(session=sess, feed_dict={x: data, y_: labels, keep_prob: 1.0})
				visualizeData(x_image_out[:,:,:,0], labels[:], 10)


	

	params.log.close()
	sess.close

### HELPER FUNCTIONS ###
# Initialize weights to normal distribution. 
def weight_variable(shape, name=None):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name)

# Initailize biase to 0.1
def bias_variable(shape, name=None):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name)

# Given data input x, uses W as a filter tensor. The input tensor is formated as [batch, in_height, in_width, in_channels].
# Defines the stride type (convolutional step size) in each dimension. 
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Performs max pooling with the size defined in ksize. 
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Create and Print Confusion Matrix
def log_confusion_matrix(y_truth, y_pred, params):
	truth_class = np.argmax(y_truth, axis=1)
	pred_class = np.argmax(y_pred, axis=1) 
	confusion = np.zeros((params.class_size, params.class_size), dtype=float)
	for num, truth_cl in enumerate(truth_class): 
		confusion[truth_cl, pred_class[num]] += 1


	print >> params.log, "Confusion Matrix:"
	print >> params.log, confusion 

	# Debug
	# print >> params.log, "Results:"
	# print >> params.log, truth_class
	# print >> params.log, pred_class
	print >> params.log, "\n\n"

# Print log header, summarizing this model. 
def print_log_header(particle_data, params):
	print >> params.log,("###### HEADER START ###### \n")
	print >> params.log,("Log file: %s")%(params.log.name)
	print >> params.log,("Graph: 2 Convolutional Layers with 2 Convolutions at 3x3")
	print >> params.log,("Equal Images Per Class: (%r)")%(EQUAL_IMAGES_PER_CLASS)
	print >> params.log,("Filter Model (%s), Saved at: %s")%("Loaded" if LOAD_FILTERS else "Not Loaded", params.filter_path)
	print >> params.log,("FC Layer Model (%s), Saved at: %s")%("Loaded" if LOAD_FC_LAYERS else "Not Loaded", params.fc_layers_path)
	print >> params.log,("Number of Classes: %d")%(params.class_size)
	print >> params.log,("Image Dimension: %dx%d")%(params.target_dim, params.target_dim)
	print >> params.log,("Training Set Size: %d")%(len(particle_data.trainlist))
	print >> params.log,("Validation Set Size: %d")%(len(particle_data.validlist))
	print >> params.log, "*** Directory Map ***"
	pprint.pprint(params.directory_map, params.log)
	print >> params.log, "*** Images Per Class ***"
	pprint.pprint(dict(particle_data.files_per_class), params.log)
	print >> params.log,("\n###### HEADER END ###### \n\n\n")

# Ca
def getLossWeights(particle_data, params):
	total_images =  sum(particle_data.files_per_class.values())
	weight = np.zeros((params.class_size))
	prob_of_class = np.zeros((params.class_size))
	for key, value in particle_data.files_per_class.iteritems():
		prob_of_class[key] = value/float(total_images)
		weight[key] = params.class_size/(prob_of_class[key] * params.class_size)

	print >> params.log,("Probabability of class based on image distribution: ")
	print >> params.log,(prob_of_class)
	print >> params.log,("Calculated Weights: ")
	print >> params.log,(weight)
	print >> params.log,("\n\n")

	return weight
	
def visualizeFilters(filter_struct):
	plt.figure(figsize = (8, 8))
	gs1 = gridspec.GridSpec(8, 8, wspace=0.0, hspace=0.0)

	for i in range(0, np.shape(filter_struct)[3]):
		ax1 = plt.subplot(gs1[i])
		plt.imshow(filter_struct[:, :,0,i])
		plt.axis('off')

	plt.savefig("./data/test.png",bbox_inches='tight')
	plt.show()

def visualizeData(data, labels, step):
	for i in range(0, np.shape(data)[0], step):
		plt.figure(i)
		plt.imshow(data[i, :, :])
		# Labeling Graph
		c = np.argmax(labels[i])
		plt.title("Class %d"%(c))
		plt.xlabel(np.transpose(labels[i]))
	plt.show()

# Command Line Sugar: Calls the main function when file executed from command-line
if __name__ == "__main__":
    main()