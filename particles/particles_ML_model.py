# Initial ML Model for Particle Classification

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import particles_batch as batch

DEBUG = 0

# ToDo: 
## Break particles_batch.py into further sub-functions
## Have next_batch return a data,labels object. 


#### FILE-SYSTEM GLOBAL VARIABLES ####
base_directory = "./data/IrisDB/"

# classes => dictionary with {class_num : list of directory names}
# classes = {
# 	0: ["IRIS-BACT"], 
# 	1: ["IRIS-CLUMPS/IRIS-WBCC", "IRIS-CLUMPS/IRIS-YSTS"], 
# 	2: ["IRIS-CRYST/IRIS-CAOX", "IRIS-CRYST/IRIS-CAPH", "IRIS-CRYST/IRIS-TPO4", "IRIS-CRYST/IRIS-URIC"], 
# 	3: ["IRIS-HYAL"], 
# 	4: ["IRIS-NHYAL/IRIS-CELL", "IRIS-NHYAL/IRIS-GRAN"], 
# 	5: ["IRIS-NSQEP/IRIS-REEP", "IRIS-NSQEP/IRIS-TREP"],
# 	6: ["IRIS-RBC"],
# 	7: ["IRIS-SPRM"],
# 	8: ["IRIS-SQEP"],
# 	9: ["IRIS-WBC"],
# }

# classes = {
# 	0: ["IRIS-BACT"], 
# 	1: ["IRIS-CLUMPS/IRIS-WBCC", "IRIS-CLUMPS/IRIS-YSTS", "IRIS-CRYST/IRIS-CAOX", "IRIS-CRYST/IRIS-CAPH", "IRIS-CRYST/IRIS-TPO4", "IRIS-CRYST/IRIS-URIC", "IRIS-HYAL", "IRIS-NHYAL/IRIS-CELL", "IRIS-NHYAL/IRIS-GRAN", "IRIS-NSQEP/IRIS-REEP", "IRIS-NSQEP/IRIS-TREP", "IRIS-RBC", "IRIS-SQEP", "IRIS-WBC"], 
# 	2: ["IRIS-SPRM"], 
# }

# classes => dictionary with {class_num : list of directory names}
classes = {
	0: ["IRIS-BACT"], 
	1: ["IRIS-RBC"],
	2: ["IRIS-SPRM"],
	3: ["IRIS-WBC"],
	4: ["IRIS-CLUMP-WBCC", "IRIS-CLUMP-YSTS", "IRIS-CRYST-CAOX", "IRIS-CRYST-CAPH", "IRIS-CRYST-TPO4", "IRIS-CRYST-URIC", "IRIS-HYAL", "IRIS-NHYAL-CELL", "IRIS-NHYAL-GRAN", "IRIS-NSQEP-REEP", "IRIS-NSQEP-TREP", "IRIS-SQEP"], 
}



def main():
	#### Variable Setup ####
	target_dim = 52
	labels_dim = len(classes)

	# Create particle data object for getting training/validation data 
	particle_data = batch.ParticleSet(base_directory, classes, target_dim)

	# Initiate Session: Interactive sessions allows for interleaving instructions that make graph and run graph. 
	sess = tf.InteractiveSession()

	# Make placeholder variables: These take the training input images, and the corresponding labels
	x = tf.placeholder(tf.float32, shape=[None, target_dim * target_dim])
	y_ = tf.placeholder(tf.float32, shape=[None, labels_dim])


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
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	# Perfomr Max Pooling: Performs max pooling on the input, striding across the data with 2x2 matrices. 
	# Essentially, we reduce each non-overlapping 2x2 matrix into a single data point. 
	# Output has shape of [batch, 26, 26, 32], where 15 is the batch size.
	h_pool1 = max_pool_2x2(h_conv1)


	# Calculate the 2nd layer. Here, we increase the channels from 32 input to 64 output. 
	#### LAYER 2: CONVOLUTIONAL LAYER ####
	# Note: Inceasing channels from 32 to 64 while reducing pixels by 4. 
	W_conv2 = weight_variable([10, 10, 32, 64])
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
	# Note: Perform softmax regression on the labels_dim classes. 
	# Note: Translation matrix that takes 1024 neuron weights as input, and converts it into labels_dim evidence outputs for each image. 
	W_fc2 = weight_variable([1024, labels_dim])
	b_fc2 = bias_variable([labels_dim])
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
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	# Operations for accuracy calculation. 
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# Initialize all variables
	# Before variables can be used within a session, they must be initialized. 
	sess.run(tf.initialize_all_variables())

	### RUN GRAPH TO TRAIN SYSTEM ###
	for i in range(100):

		# Obtain training data
		# data => 
		# labels => [batch, len(classes)]
		data, labels = particle_data.next_batch(50)

		# At every xth batch, log the result. 
		if i%50 == 0:
			# Evaluates the accuracy (doesn't drop nodes during evaluatoin)
			train_accuracy = accuracy.eval(feed_dict={x: data, y_: labels, keep_prob: 1.0})
			print("step %d, training accuracy %g"%(i, train_accuracy))

			if (DEBUG):
				x_image_out = x_image.eval(session=sess, feed_dict={x: data, y_: labels, keep_prob: 1.0})
				visualizeData(x_image_out[:,:,:,0], labels[:], 5)


		#Runs a single train step with a single batch using a keep probability of 0.5
		train_step.run(feed_dict={x: data, y_: labels, keep_prob: 0.5})


	### FINAL SUMMARY ###
	data, labels = particle_data.next_batch(50)
	#Prints the final accuracy
	print("test accuracy %g"%accuracy.eval(feed_dict={x: data, y_: labels, keep_prob: 1.0}))
	# Prints final weights ( by fethcing them out of tensorflow graph)
	W_fc2_out =  W_fc2.eval(session=sess)
	print W_fc2_out


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
		plt.title(classes[c])
		plt.xlabel(np.transpose(labels[i]))
	plt.show()

# Command Line Sugar: Calls the main function when file executed from command-line
if __name__ == "__main__":
    main()