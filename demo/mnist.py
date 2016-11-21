# MNIST Categorization

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


### IMPORT DATASET ###
# 55,000 data points of training data (train.images)
# 10,000 data points of testing data (test.images)
# 5,000 data points of validation data. 
# mnist.train.image: Each image is 28px by 28px (or an array of 784)
# mnist.train.images: An array of [55000, 784] (with the pixels represented as float from 0 to 1).
# mnist.train.lables: An array of [55000, 10] of floats. Each label array is all 0s except with a 1 corresponding to image category. 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

### CREATE SYMBOLIC VARIABLES ###
# Placeholder symbolic variable. Placeholder represents a value that we will input into the graph. 
# We represent the input as 2D array with undetermined rows (images)
x = tf.placeholder(tf.float32, [None, 784])

# Variable is a symoblic variable that can be modified and is within tensorflow's graph of interacting operations.
# The model parameters are generally variables. 
# These variables are initialized to 0, but can be initialized to arbitrary inputs. 
# Create W: For each category, each pixel gets a different weight (dependent on back-propogation)
W = tf.Variable(tf.zeros([784, 10]))
# Create b: Each category has a constant bias offset to take into account things like the absolute propobablity of each category (prior).
b = tf.Variable(tf.zeros([10]))

# Save/restore W and b. 
saver = tf.train.Saver({"my_W": W, "my_b": b})


### DEFINE MODEL ###
# x and W were defined so that size(x*W) = [None,10]. Essentially, we get the evidence across the 10 categories. 
# Note: Can also use softmax_cross_entropy_with_logits directly, which apperantly is more numerically stable.  
y = tf.nn.softmax(tf.matmul(x, W) + b)

### TRAIN MODEL ###
# Placeholder variable for the labels for each MNIST image. 
y_ = tf.placeholder(tf.float32, [None, 10])

# Define the cost or lost of each iteration of your prediction. 
# Implement cross-entropy function where y_  is the real distribution and y i the predicted distribution. 
# Reduction_indices of [1] indicates that we reduce across columns. 
# Basc arithmetic signs lead to element manipulation. You need special functions for matrix multiplication. 
# The tf.reduce_mean computes the average of all the samples in the batch. So, the cross-entropy is computed for each input, and finally averaged across the batch's input. 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.Print(cross_entropy, [cross_entropy], "My Cross Entropy: ")

# Implement backpropogation with Gradient Deescent (with a stepsize of 0.5).
# Gradient Descent finds the optimal implementation by iterating. We pass in the cost function (cross_entropy), and the list of Weights to be updated.  
# Essentially, given the results captured in cross_entropy, we updated the Variables W and b (default) based on a gradient descent optimization. 
# Gradietn descent just shifts each variable slightly in the direction of optimal results. 
# This step does: 1) adds operations to graph that implement backpropogation and gradient descent and 2) gives a operation that does a single step of gradient descent training. 
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Defines the operation to initialize all variables we are creating (essentially inidializes the nodes of the graph)
init = tf.initialize_all_variables()

### RUN A SESSION ###
# A Session is a class that creates an environment that allows graphs to be run (operations to be executed). 
# Important to close session once completed (to release those rescources). 
sess = tf.Session()
# Runs operations (that we previously defined) by fetching values obtained by running parts of the graph. 
# Here, we just initialize the variables. 
sess.run(init)

# Restores variable W and b (stored in model.ckpt). Since these are the only variables, we don't need to initialize variables seperately/ 
#saver.restore(sess, "./data/mnist1_model.ckpt")


for i in range(100):
	# batch_xs is the iimage matrix with 784 pixels, and 100 images. And, batch_ys is the corresponding labeled output. 
	# next_patch defined in tensorflow/learn/python/learn/datasets/data/mnist.py. Essentially grabs random images for training. 
	# Since we take a random batch, we are implementing stochastic training. 
	batch_xs, batch_ys = mnist.train.next_batch(100)
	# Runs the training step operationg. In order to run train_step, we need to calculate the cross-entropy, which propogates through an entire graph. 
	# feed_dic initializs x and y_ in the graph (essentially sets up a placeholder variable).
	# We send multiple images at once. 
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	# argmax returns the index with the max entry across the column dimension for the entire matrix. 
	# tf.equal returns the element-wise comparison of maximum indexes. 
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	# We cast the correct_prediction to a floating point, and then take the mean of the array. 
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	# After the the model has been trained by running the train_step, we pass through test images/lables. 
	# We return the accuracy.  
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


### FINAL SUMMARY ###
# Obtain final bias and weights (through tensorflow graph)
W = tf.Print(W, [W], "Weights: ", summarize = 100)
b = tf.Print(b, [b], "Bias: ", summarize = 10)
sess.run(W)
sess.run(b)

# Save Session
save_path = saver.save(sess, "./data/mnist1_model.ckpt")
print("Model saved in file: %s" % save_path)

# Convert weights into a numpy array (and use this for visualization)
# W_all is a [784,10] nparray. 
W_all = W.eval(session=sess)
sess.close()

fig = plt.figure()
for i in range(10):
	W_i = W_all[:,i].reshape((28,28))
	fig.add_subplot(2,5,i+1)
	plt.imshow(W_i)
	plt.axis('off')

plt.show()
