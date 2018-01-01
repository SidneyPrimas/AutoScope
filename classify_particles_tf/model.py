import tensorflow as tf
import numpy as np

# Homebrew libraries
import utility_functions

class ParticleDetectionModel(object):

	def __init__(self, params):

		self.params = params


		# Define tensor shapes
		self.image_tensor_shape = [None, self.params.target_dim * self.params.target_dim]
		self.batch_tensor_shape = [None, self.params.class_size]
		self.keep_prob = tf.placeholder(tf.float32)

		# Make placeholder variables: These take the training input images, and the corresponding labels
		self.images_placeholder = tf.placeholder(tf.float32, shape=self.image_tensor_shape)
		self.labels_placeholder = tf.placeholder(tf.float32, shape=self.batch_tensor_shape)

		# Build graph: model and training infrastructure 
		self.prediction, saver_objects = self.build_model(self.images_placeholder)
		self.saver_filters, self.saver_fc_layers = saver_objects
		self.loss = self.build_loss_system(self.prediction, self.labels_placeholder, weights=self.params.weights)
		self.train_op = self.build_train_system(self.loss)
		self.accuracy = self.build_accuracy_system(self.prediction, self.labels_placeholder)


	def build_model(self, x_input):

		# Reshape data: Reshapes each input x from a 2-D Tensor into a 4D Tensor [batch, image_height, image_width, channel]. 
		x_image = tf.reshape(x_input, [-1, self.params.target_dim, self.params.target_dim, self.params.channels], name= "initial_reshape_op")

		#### Block 1: CONVOLUTIONAL LAYER ####
		# Create variables
		block1_filters = 32
		W_conv1_a = self.weight_variable([3, 3, self.params.channels, block1_filters], name='W_conv1_a')
		b_conv1_a = self.bias_variable([block1_filters], name='b_conv1_a')
		W_conv1_b = self.weight_variable([3, 3, block1_filters, block1_filters], name='W_conv1_b')
		b_conv1_b = self.bias_variable([block1_filters], name='b_conv1_b')

		# Perform block 1 functions
		h_conv1_a = tf.nn.relu(self.conv2d(x_image, W_conv1_a) + b_conv1_a)
		h_conv1_b = tf.nn.relu(self.conv2d(h_conv1_a, W_conv1_b) + b_conv1_b)
		h_pool1 = self.max_pool_2x2(h_conv1_b)

		#### Block 2: CONVOLUTIONAL LAYER ####
		# Create variables
		block2_filters = 64
		W_conv2_a = self.weight_variable([3, 3, block1_filters, block2_filters], name='W_conv2_a')
		b_conv2_a = self.bias_variable([block2_filters], name='b_conv2_a')
		W_conv2_b = self.weight_variable([3, 3, block2_filters, block2_filters], name='W_conv2_b')
		b_conv2_b = self.bias_variable([block2_filters], name='b_conv2_b')

		# Perform block 2 functions
		h_conv2_a = tf.nn.relu(self.conv2d(h_pool1, W_conv2_a) + b_conv2_a)
		h_conv2_b = tf.nn.relu(self.conv2d(h_conv2_a, W_conv2_b) + b_conv2_b)
		h_pool2 = self.max_pool_2x2(h_conv2_b)

		#### Block 3: Fully Connected Layer1 ####
		# Create variables
		block3_weights = 1024
		W_fc1 = self.weight_variable([self.params.target_dim/4 * self.params.target_dim/4 * block2_filters, block3_weights], name='W_fc1')
		b_fc1 = self.bias_variable([block3_weights], name='b_fc1')

		# Perform block 3 functions
		h_pool2_flat = tf.reshape(h_pool2, [-1, self.params.target_dim/4 * self.params.target_dim/4 * block2_filters])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
		h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)


		### Block 4: READOUT LAYER  ####
		# Create variables
		W_fc2 = self.weight_variable([block3_weights, self.params.class_size], name='W_fc2')
		b_fc2 = self.bias_variable([self.params.class_size], name='b_fc2')
		# Perform block 4 functions
		y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

		# Create saver opject. 
		saver_filters = tf.train.Saver([W_conv1_a, b_conv1_a, W_conv1_b, b_conv1_b, W_conv2_a, b_conv2_a, W_conv2_b, b_conv2_b])
		saver_fc_layers = tf.train.Saver([W_fc1, b_fc1, W_fc2, b_fc2])
		saver_object_list = [saver_filters, saver_fc_layers]

		return y_conv, saver_object_list


	def build_loss_system(self, prediction, truth, weights=None):

		# Created a weighted loss function
		if weights == None:
			cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=truth, logits=prediction))
			
		# Create a regular, non-weighted loss function (according to the proportion of images)
		else: 
			pos_weight = tf.constant(weights, dtype=tf.float32)
			cross_entropy_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=truth, logits=prediction, pos_weight=pos_weight))

		return cross_entropy_loss

	def build_train_system(self, loss):
		"""Uses Adam optimizer to update all variables in our system when training."""
		return tf.train.AdamOptimizer(self.params.learning_rate).minimize(loss)


	def build_accuracy_system(self, prediction, truth):
		"""Operations for accuracy calculation.""" 
		correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(truth,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return accuracy


	def train_forever(self, session, data):

		count = 0 
		while True: 
			count+=1
			for epoch in range(self.params.num_epochs): 
				self.train_single_epoch(session, data)

				self.val_single_epoch(session, data)

			self.save_graph_variables(session, count)


	def train_single_epoch(self, session, data):

		accuracy_all = []
		for batch in range(self.params.batches_per_epoch_train):

			self.params.image_trained_count += self.params.train_batch_size

			# Obtain validation data
			image_data, labels = data.next_batch(self.params.train_batch_size, validation=False, per_class_order=self.params.equal_images_per_class)
			input_feed = {
				self.images_placeholder: image_data,
				self.labels_placeholder: labels, 
				self.keep_prob: self.params.dropout
			}

			output_feed = [
				self.train_op,
				self.accuracy, 
			]

			_, accuracy_value = session.run(output_feed, feed_dict=input_feed)
			accuracy_all.append(accuracy_value)

		# Calculate metrics
		accuracy_mean = np.mean(accuracy_all)
		# Print/Visualize Results
		print >> self.params.log,("Results for a epoch...")
		print >> self.params.log,("During Training Accuracy: %1.3f "%(accuracy_mean))


	def val_single_epoch(self, session, data):

		# Track overall label metrics
		all_truth = None
		all_pred = None
		accuracy_all = []
		loss_all = []

		for batch in range(self.params.batches_per_epoch_val):
			# Obtain training data
			image_data, labels = data.next_batch(self.params.validation_batch_size, validation=True, per_class_order=self.params.equal_images_per_class)
			input_feed = {
				self.images_placeholder: image_data,
				self.labels_placeholder: labels, 
				self.keep_prob: 1.0
			}

			output_feed = [
				self.accuracy, 
				self.loss, 
				self.labels_placeholder, 
				self.prediction
			]
			# Important: Do not run optimizer operation (so no training)
			accuracy_value, loss_value, truth_values, pred_values = session.run(output_feed,feed_dict=input_feed)

			# Append to overall metrics
			accuracy_all.append(accuracy_value)
			loss_all.append(loss_value)
			if all_truth is None: 
				all_truth = truth_values
				all_pred = pred_values
			else: 
				all_truth = np.append(all_truth, truth_values, axis=0)
				all_pred = np.append(all_pred, pred_values, axis=0)


		# Calculate metrics
		loss_mean = np.mean(loss_all)
		accuracy_mean = np.mean(accuracy_all)
		confusion = utility_functions.get_confusion_matrix(all_truth, all_pred)


		# Print/Visualize Results
		print >> self.params.log,("Images Trained: %d, Batch loss: %2.6f, Training accuracy: %1.3f "%(self.params.image_trained_count, loss_mean, accuracy_mean))
		print >> self.params.log, "Confusion Matrix:"
		print >> self.params.log, confusion 
		print >> self.params.log, "\n\n"

		# Visualize filters (if enabled)
		if (self.params.visualize_filters):
			self.get_filters(session, image_data, labels)

		# Visualze input images with labels (if enabled)
		if (self.params.debug):
			self.get_input_images(session, image_data, labels)




	def get_filters(self, session, image_data, labels):

		input_feed = {
			self.images_placeholder: image_data,
			self.labels_placeholder: labels, 
			self.keep_prob: 1.0
		}

		W_conv1_b_ref = tf.get_default_graph().get_tensor_by_name("W_conv1_b:0")
		W_conv2_b_ref = tf.get_default_graph().get_tensor_by_name("W_conv2_b:0")
		output_feed = [
			W_conv1_b_ref,
			W_conv2_b_ref
		]

		filters1_output, filters2_output = session.run(output_feed, feed_dict=input_feed)
		utility_functions.visualize_filters(filters1_output)
		utility_functions.visualize_filters(filters2_output)


	def get_input_images(self, session, image_data, labels):
			x_image = tf.get_default_graph().get_operation_by_name("initial_reshape_op").outputs[0]

			input_feed = {
				self.images_placeholder: image_data,
				self.labels_placeholder: labels, 
				self.keep_prob: 1.0
			}

			x_image_values = session.run(x_image, feed_dict=input_feed)
			utility_functions.visualize_image_data(x_image_values[:,:,:,0], labels[:], 2)
	
	def restore_graph_variables(self, session):
		"""
		Restore variables (weights) from previous training sessions. 
		"""

		if (self.params.filter_load_path is not None): 
			self.saver_filters.restore(session, self.params.filter_load_path)
			print >> self.params.log,("Restored Filters from: %s\n")%(self.params.filter_load_path)


		if (self.params.fc_layers_load_path is not None): 
			self.saver_fc_layers.restore(session, self.params.fc_layers_load_path)
			print >> self.params.log,("Restored Fully-Connected Layers from: %s\n")%(self.params.fc_layers_load_path)


	def save_graph_variables(self, session, count):
		""" Save variables (save filters and fully connected layers seperately) """

		if (self.params.filter_save_path is not None): 
			save_path = self.params.filter_save_path 
			self.saver_filters.save(session, save_path)
			print >> self.params.log,("Fitler Models saved in file: %s" % save_path)

		if (self.params.fc_layers_save_path is not None): 
			save_path = self.params.fc_layers_save_path 
			self.saver_fc_layers.save(session, save_path)
			print >> self.params.log,("Fully Connected Models saved in file: %s" % save_path)
			


	###### HELPER FUNCTIONS FOR MODEL LAYERS ####
	def weight_variable(self, shape, name=None):
		"""Initialize weights to normal distribution. """
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial, name=name)

	def bias_variable(self, shape, name=None):
		"""Initailize biase to 0.1"""
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial, name=name)

	def conv2d(self, x, W):
		"""
		Given data input x, uses W as a filter tensor. The input tensor is formated as [batch, in_height, in_width, in_channels].
		Defines the stride type (convolutional step size) in each dimension. 
		"""
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(self, x):
		""" Performs max pooling with the size defined in ksize. """
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	
