# MNIST Categorization for Experts

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# mnist is a class that provides functions that store and iterate over training data. 
from tensorflow.examples.tutorials.mnist import input_data

### FUNCTIONS ###
# Initialize weights to normal distribution. 
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# Initailize biase to 0.1
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Given data input x, uses W as a filter tensor. The input tensor is formated as [batch, in_height, in_width, in_channels].
# The stride type (convolutional step size) in each dimension. 
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Performs max pooling with the size defined in ksize. 
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


### MAIN CODE ###

# Import the MNIST data set with the mnist class (as defined above)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# We will be working with an interactive session (allows you to interleave instructions that make a graph with ones that run it). 
sess = tf.InteractiveSession()

# Build a placeholder variable that will take our training input. y_ is the lablels. 
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Reshapes each input x from a 2-D Tensor into a 4D Tensor [batch, image_height, image_width, channel]. 
# -1 used to indicate that this dimension selected so total size remains constant. In this case, just transfer the None. 
x_image = tf.reshape(x, [-1,28,28,1])


# Building the first convolutional layer. 
# The weight variable includes a 5x5 patch (the filter). The 1 is the number of input channels, and the 32 is the number of output channels. 
# 32 is an arbitrary output. 
W_conv1 = weight_variable([5, 5, 1, 32])
# We have a bias variable for each channel
b_conv1 = bias_variable([32])

# Convolve x_image with W_conv1 filter, and add the offset. Then, take relu of each element in the output. 
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# Performs max pooling on the input, striding across 2x2 matrices. Essentially, we reduce 2x2 matrix into a single data pont. 
# Dimensinos of [50, 14, 14, 32], where 15 is the batch size.
h_pool1 = max_pool_2x2(h_conv1)

# Calculate the 2nd layer. Here, we increase the channels from 32 input to 64 output. 
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer: Should allow processing on entire image. 
# Since we have had 2 max-pooling layers condesing 4 pixels into one, we now have a 7x7 image. Each pixel has 64 channels. 
# Essentially, we linearize the data from each pixel (7*7 pixels with each pixels having 64 channels).
# Then we create 1024 neurons, which represents the output number. 
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# Reshpae h_pool2 into [batch, linearize (im_pixel*channels)]. 
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# matmul gives us [batch, 1024]. Or, for each input image, we get 1024 outputs. 
# Here we do another calculation of evidence of the convolutional layers so far. 
# Essentially, the convolutional layers perform feature detection. And, here we turn the features into evidence, and pass the evidence through ReLU. 
# Feature detection is a learned process. 
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Add a dropout layer. 
# The placeholder allows us to insert the dropout probability during training, and remove it during testing. 
keep_prob = tf.placeholder(tf.float32)
# dropout drops elements. For elements it keeps, it scales by 1/keep_prob. And sets other elements to 0. This should keep the expected sume the same. 
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer that performs softmax regression. 
# Translation matrix that takes 1024 neuron weights as input, and converts it into 10 outputs for each image. 
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# Implement the regression model that converts inputs (x) into probably outcomes (y) using W and b as trained variables. 
# y_conv is a size of [batch, label #s]. This will be used for softmax, and the loss function. 
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# Defines the loss function. We try to train W and b in order to minimize loss across all training inputs. 
# softmax_cross_entropy_with_logits: applies softmax across unnormalized data, and sums across all classes. 
# tf.reduce_mean: Computes the mean of the batch (so that we only update once per batch). 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

# Uses Adam optimizer to update all variables in our system
# When we run train_step, we updated all W and b. 
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Operations for accuracy calculation. 
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Before variables can be used within a session, they must be initialized. 
sess.run(tf.initialize_all_variables())

for i in range(10):
	# batch is a 2D tensor with batch_xs and batch_ys
  batch = mnist.train.next_batch(50)

  # Includes logging at every 100th batch. 
  if i%100 == 0:
  	# Evaluates the accuracy by not dropping any nodes. 
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))


  #Runs a single train step with a single batch usign a keep probability of 0.2
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#Prints the final accuracy
print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


### FINAL SUMMARY ###
batch = mnist.train.next_batch(50)
# Obtain final bias and weights (through tensorflow graph)
h_pool1_out =  h_pool1.eval(session=sess, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
print h_pool1_out.shape


sess.close


