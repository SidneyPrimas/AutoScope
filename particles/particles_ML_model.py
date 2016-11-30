# Initial ML Model for Particle Classification

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import particles_batch as batch

# Global Variables
DEBUG = False
LOAD_ENABLE = True
TRAINING = True

# ToDo: 
## Break particles_batch.py into further sub-functions
## Have next_batch return a data,labels object. 
## Better to have filelist in memory (or constantly call gob)
## Figure out in what proportions we should train our network with
## Automatically restore last saved variables
## Is it bad to use a random number (issues with not being a real random number)
## Create wrapper function that just creates the basic neural network (this allows us seperate validation and training)
## Optimization: When call the accuracy values, call them all at once (instead of re-running the graph)
## When print >> log,ing out per-class accuracies, find a more elegent solution than adding one to class number.
## When print >> log,ing out per-class accuracies, find an automated way to pretty print >> log,
## How many filters per layer


#### FILE-SYSTEM GLOBAL VARIABLES ####
base_directory = "./data/IrisDB/"

# directory_map => dictionary with {folder_path : class number folder belong to}
directroy_map = {
	"IRIS-BACT": 0, 
	"IRIS-RBC": 1, 
	"IRIS-SPRM": 3, 
	"IRIS-WBC": 2, 
	"IRIS-CLUMP-WBCC": 2,
	"IRIS-CLUMP-YSTS": 3,
	"IRIS-CRYST-CAOX": 3,
	"IRIS-CRYST-CAPH": 3,
	"IRIS-CRYST-TPO4": 3,
	"IRIS-CRYST-URIC": 3, 
	"IRIS-HYAL": 3,
	"IRIS-NHYAL-CELL": 3,
	"IRIS-NHYAL-GRAN": 3,
	"IRIS-NSQEP-REEP": 3,
	"IRIS-NSQEP-TREP": 3,
	"IRIS-SQEP": 3, 
}


def main():
	### Configurable Variables ###
	training_iters = 10
	train_batch_size = 100
	validation_batch_size = 500

	step_display = 10
	step_save = 10
	learning_rate = 1e-4
	dropout = 0.5
	path_save = "./data/particle_model"
	path_load = "./data/particle_model"

	#### Variable Setup ####
	target_dim = 52
	class_size = 4
	validation_proportion = 0.1

	# Create particle data object for getting training/validation data 
	particle_data = batch.ParticleSet(base_directory, directroy_map, class_size, target_dim, validation_proportion)

	# Initiate Session: Interactive sessions allows for interleaving instructions that make graph and run graph. 
	sess = tf.InteractiveSession()

	# Make placeholder variables: These take the training input images, and the corresponding labels
	x = tf.placeholder(tf.float32, shape=[None, target_dim * target_dim])
	y_ = tf.placeholder(tf.float32, shape=[None, class_size])


	# Reshape data: Reshapes each input x from a 2-D Tensor into a 4D Tensor [batch, image_height, image_width, channel]. 
	# Note: -1 used to indicate that this dimension selected so total size remains constant. In this case, just transfer the None. 
	x_image = tf.reshape(x, [-1, target_dim, target_dim, 1])

	#### LAYER 1: CONVOLUTIONAL LAYER ####
	# Note: Increase channels from 1 to 32 while reducing pixels by 4. 
	# Define the Filter: The weight variable is a 5x5 kernel (the filter). 
	# The convolutional layer transforms 1 input channel (greyscale) to 32 output channels (32 is arbitrary)
	W_conv1 = weight_variable([5, 5, 1, 32])
	# Define the bias for each channel. 
	b_conv1 = bias_variable([32])

	# Perform Convolution: Convolve x_image with W_conv1 filter, and add the offset. 
	# Then, for each channel of each pixel, take the ReLU. 
	# Size of h_conv1: [batch, 52, 52, 32]
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	# Perfomr Max Pooling: Performs max pooling on the input, striding across the data with 2x2 matrices. 
	# Essentially, we reduce each non-overlapping 2x2 matrix into a single data point. 
	# Output has shape of [batch, 26, 26, 32], where 15 is the batch size.
	h_pool1 = max_pool_2x2(h_conv1)


	# Calculate the 2nd layer. Here, we increase the channels from 32 input to 64 output. 
	#### LAYER 2: CONVOLUTIONAL LAYER ####
	# Note: Inceasing channels from 32 to 64 while reducing pixels by 4. 
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	#### LAYER 3: DENSELEY CONNECTED LAYER ####
	# Note: A denseley connected layer starts using a regression model to translate data into evidence.  
	# Note: Densely Connected Layer allows processing on entire input. 
	# Notes on input: With 2 max-pooling layers condensing 4-px to 1px, we now have the following shape [batch, 13, 13, 64]
	# First, we linearize all input data, including the 64 channels across each pixel. 
	# Then we create 1024 neurons, which represents the output. 
	W_fc1 = weight_variable([13 * 13 * 64, 1024])
	b_fc1 = bias_variable([1024])

	# Reshpae h_pool2 into [batch, linearize (im_pixel*channels)]. 
	# In order to allow h_pool2 to be an input in the correct format, we must reshape it. 
	h_pool2_flat = tf.reshape(h_pool2, [-1, 13 * 13 * 64])

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
	# Note: For nodes it keeps, it scales by 1/keep_prob. And sets other elements to 0. This should keep the expected sume the same. 
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


	### Layer 5: READOUT LAYER (REGRESSION LAYER) ####
	# Note: Perform softmax regression on the class_size classes. 
	# Note: Translation matrix that takes 1024 neuron weights as input, and converts it into class_size evidence outputs for each image. 
	W_fc2 = weight_variable([1024, class_size])
	b_fc2 = bias_variable([class_size])
	# Note: Implement the regression model that converts inputs (h_fc1_drop) into probably outcomes (y_conv) using W and b as trained variables. 
	# y_conv is a size of [batch, label #s]. Use for softmax, and the loss function. 
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


	### SETUP LOSS FUNCTION AND TRAINING IN GRAPH ###
	# Defines the loss function: Train all W and b in order to minimize loss across all training inputs. 
	# Note on softmax_cross_entropy_with_logits: applies softmax across unnormalized data, and sums across all classes. 
	# Notes on tf.reduce_mean: Computes the mean of the batch (so that we only update the variables once per batch). 
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

	# Uses Adam optimizer to update all variables in our system
	# When we run train_step, we updated all W and b. 
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

	# Operations for accuracy calculation. 
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	# Breakdown of inaccuracies across classes
	# Identifies the actual class when a prediction was wrong. 
	# Important: We add 1 to argmax (increasing the class size by one) so that the 0 from an accurate prediction is not conflicting with class 0. 
	ground_truth = tf.cast(tf.not_equal(tf.argmax(y_conv,1), tf.argmax(y_,1)), tf.int64) * (tf.argmax(y_,1) + 1)
	# Identifies the class we mistakenly selected when a prediction is wrong. 
	inaccurate_pred = tf.cast(tf.not_equal(tf.argmax(y_conv,1), tf.argmax(y_,1)), tf.int64) * (tf.argmax(y_conv,1) + 1)

	### INITIALIZE ALL VARIABLES ###
	# Create saver opject. 
	# Note: Since no variables are explicitly specified, saver stores all trainable variables. 
	# ToDo: Include keep_checkpoint_every_n_hours to only keep checkpoint every n hours. 
	saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

	# Before variables can be used within a session, they must be initialized. 
	if (not LOAD_ENABLE):
		sess.run(tf.initialize_all_variables())

	# Restores all trained variables (stored in model.ckpt). 
	# Since we restore all variables from the previous session, we don't need to initialize any variables. 
	if (LOAD_ENABLE): 
		saver.restore(sess, path_load)

	### RUN GRAPH TO TRAIN SYSTEM ###
	log=open('./log/log', 'a', 1)
	for i in range(training_iters):

		# Train
		#Runs a single train step with a single batch using a keep probability of 0.5
		if (TRAINING):
			# Obtain training data
			data, labels = particle_data.next_batch(train_batch_size)
			train_step.run(feed_dict={x: data, y_: labels, keep_prob: dropout})

		# Save model 
		# ToDo: Implement better saving mechanism
		if (i%step_save == 0) and TRAINING: 
			# Save trainted model
			# Saves all trainable variables in this session. 
			save_path = saver.save(sess, path_save)
			print >> log,("Model saved in file: %s" % save_path)


		if (i%step_display == 0) or (not TRAINING):

			# Obtain training data
			data, labels = particle_data.next_batch(validation_batch_size, validation=True)

			# Evaluates the accuracy and cross_entropy (doesn't drop nodes during evaluation)
			# Obtain all values from graph in single session. 
			train_accuracy, train_loss, truth_train, inaccurate_train = sess.run([accuracy, cross_entropy, ground_truth, inaccurate_pred],feed_dict={x: data, y_: labels, keep_prob: 1.0})
			print >> log,("step: %d, batch loss: %2.6f, training accuracy: %1.3f "%(i, train_loss, train_accuracy))
			

			# Note: GT_Classes are sorted 
			truth_classes, truth_counts = np.unique(truth_train, return_counts=True)
			truth_perc = 100*truth_counts/float(len(truth_train))
			print >> log, "When wrong, correct solution:",
			for n in range(len(truth_classes)): 
				if (truth_classes[n] == 0):
					print >> log,("\t Accurate: %2.2f, "%(truth_perc[n])),
				else: 
					# Subract 1 from n since needed to previously increase class by one to ensure no conflict with 0 from accurate prediction. 
					c = truth_classes[n] - 1
					print >> log,("Class %d: %2.2f, "%(c, truth_perc[n])),
			print >> log, ""

			inaccurate_classes, inaccurate_counts = np.unique(inaccurate_train, return_counts=True)
			inaccurate_perc = 100*inaccurate_counts/float(len(inaccurate_train))
			print >> log, "When wrong, incorrect guess:",
			for n in range(len(inaccurate_classes)): 
				if (inaccurate_classes[n] == 0):
					print >> log,("\t Accurate: %2.2f, "%(inaccurate_perc[n])),
				else: 
					# Subract 1 from n since needed to previously increase class by one to ensure no conflict with 0 from accurate prediction. 
					c = inaccurate_classes[n] - 1
					print >> log,("Class %d: %2.2f, "%(c, inaccurate_perc[n])),
			print >> log, "\n"

			if (DEBUG):
				x_image_out = x_image.eval(session=sess, feed_dict={x: data, y_: labels, keep_prob: 1.0})
				visualizeData(x_image_out[:,:,:,0], labels[:], 10)


	

	###  FINAL SUMMARY ###
	data, labels = particle_data.next_batch(validation_batch_size, validation=True)
	#Print the final accuracy
	print >> log,("Final Accuracy %1.3f"%accuracy.eval(feed_dict={x: data, y_: labels, keep_prob: 1.0}))
	save_path = saver.save(sess, path_save)
	print >> log,("Final Model saved in file: %s" % save_path)

	log.close()
	sess.close

### HELPER FUNCTIONS ###
# Initialize weights to normal distribution. 
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

# Initailize biase to 0.1
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# Given data input x, uses W as a filter tensor. The input tensor is formated as [batch, in_height, in_width, in_channels].
# Defines the stride type (convolutional step size) in each dimension. 
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Performs max pooling with the size defined in ksize. 
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


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