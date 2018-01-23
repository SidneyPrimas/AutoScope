# Import basic libraries
import tensorflow as tf
import numpy as np
import h5py  

# Import keras libraries
from tensorflow.python.keras.models import Model # Allows to build more complex models than Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, GlobalAveragePooling2D, Dropout # Import custom layers. 

def base_model(input_shape, base_weights, classes): 
	"""
	Configuration: data_format => channel_last
	Args: 
	input_shape: shape of image, including the channel. 
	Return: 
	An instantiated model.
	"""
	img_input = Input(shape=input_shape)

	# Block 1
	# When using Conv2D as first layer, need to correctly connect the layer to input with input_shape argument. 
	x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
	x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
	f1 = x

	# Block 2
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
	f2 = x

	# Fully-connected layers (needed in order to load original VGG16 imagenet weights). 
	x = Flatten(name='flatten')(x)
	x = Dense(1024, activation='relu', name='fc1')(x)
	x = Dropout(0.5)(x) # Automatically disabled during validation
	x = Dense(classes, activation='softmax', name='predictions')(x) # Since loading pre-trained weights, need VGG classes. 

	model  = Model(img_input, x, name = "base_model")

	return model

def VGG16_with_custom_FC_average(input_shape, base_weights, classes): 
	"""
	Returns VGG16 model with custom FCs (using average instead of flatten)
	Notes: The custom FC reduces the number of weights that need to be learned for faster training. 
	Args: 
	Args: 
	input_shape: shape of image, including the channel. 
	base_weights: Path to file with pre-trained weights. 
	include_top: If true, then use original weights for convolutional layers, and add new FCs to the top. 
	Return: 
	An instantiated model.
	"""

	vgg16_base = VGG16_original(input_shape = input_shape, base_weights = base_weights, include_top = False)
	# Fully-connected layers (needed in order to load original VGG16 imagenet weights). 
	x = vgg16_base.output
	x = GlobalAveragePooling2D()(x) #Instead of just flattening, reduces the features even further for faster training.
	x = Dense(1024, activation='relu', name='fc1')(x) # Single fully-connected layer. 
	x = Dropout(0.5)(x)
	predictions = Dense(classes, activation='softmax', name='predictions')(x) # Output layer
	
	return Model(inputs=vgg16_base.input, outputs = predictions, name = "VGG16_with_custom_FC_average")


def VGG16_with_custom_FC_flatten(input_shape, base_weights, classes): 
	"""
	Returns VGG16 model with custom FCs (use flatten instead of average)
	Notes: The custom FC reduces the number of weights that need to be learned for faster training. 
	Args: 
	Args: 
	input_shape: shape of image, including the channel. 
	base_weights: Path to file with pre-trained weights. 
	include_top: If true, then use original weights for convolutional layers, and add new FCs to the top. 
	Return: 
	An instantiated model.
	"""

	vgg16_base = VGG16_original(input_shape = input_shape, base_weights = base_weights, include_top = False)
	# Fully-connected layers (needed in order to load original VGG16 imagenet weights). 
	x = vgg16_base.output
	x = Flatten()(x) #Instead of just flattening, reduces the features even further for faster training.
	x = Dense(1024, activation='relu', name='fc1')(x) # Single fully-connected layer. 
	x = Dropout(0.5)(x)
	predictions = Dense(classes, activation='softmax', name='predictions')(x) # Output layer
	
	return Model(inputs=vgg16_base.input, outputs = predictions, name = "VGG16_with_custom_FC_flatten")

def VGG16_bottom3_layers_custom_FC_flatten(input_shape, base_weights, classes): 
	"""
	Keras model with the first 3 layers of VGG16. Always preload the base_weights. 
	"""
	img_input = Input(shape=input_shape)

	# Block 1
	# When using Conv2D as first layer, need to correctly connect the layer to input with input_shape argument. 
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
	f1 = x

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
	f2 = x

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
	f3 = x

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
	f4 = x

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
	f5 = x

	# Custom fully connected layer
	x = Flatten(name='flatten')(f3)
	x = Dense(1024, activation='relu', name='fc1')(x)
	x = Dropout(0.5)(x)
	predictions = Dense(classes, activation='softmax', name='predictions')(x)


	# Load imagenet weights
	vgg  = Model(inputs=img_input, outputs = f5, name = "VGG16_decapitated")
	vgg.load_weights(base_weights)


	output_model = Model(inputs=img_input, outputs = predictions, name = "VGG16_bottom3_layers_custom_FC_flatten")
	return output_model


def VGG16_bottom3_layers_custom_FC_average(input_shape, base_weights, classes): 
	"""
	Keras model with the first 3 layers of VGG16. Always preload the base_weights. 
	"""
	img_input = Input(shape=input_shape)

	# Block 1
	# When using Conv2D as first layer, need to correctly connect the layer to input with input_shape argument. 
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
	f1 = x

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
	f2 = x

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
	f3 = x

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
	f4 = x

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
	f5 = x

	# Custom fully connected layer
	x = GlobalAveragePooling2D(name='flatten')(f3)
	x = Dense(1024, activation='relu', name='fc1')(x)
	x = Dropout(0.5)(x)
	predictions = Dense(classes, activation='softmax', name='predictions')(x)


	# Load imagenet weights
	vgg  = Model(inputs=img_input, outputs = f5, name = "VGG16_decapitated")
	vgg.load_weights(base_weights)


	output_model = Model(inputs=img_input, outputs = predictions, name = "VGG16_bottom3_layers_custom_FC_average")
	return output_model


def VGG16_bottom2_layers_custom_FC_average(input_shape, base_weights, classes): 
	"""
	Keras model with the first 2layers of VGG16. Always preload the base_weights. 
	"""
	img_input = Input(shape=input_shape)

	# Block 1
	# When using Conv2D as first layer, need to correctly connect the layer to input with input_shape argument. 
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
	f1 = x

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
	f2 = x

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
	f3 = x

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
	f4 = x

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
	f5 = x

	# Custom fully connected layer
	x = GlobalAveragePooling2D(name='flatten')(f2)
	x = Dense(1024, activation='relu', name='fc1')(x)
	x = Dropout(0.5)(x)
	predictions = Dense(classes, activation='softmax', name='predictions')(x)


	# Load imagenet weights
	vgg  = Model(inputs=img_input, outputs = f5, name = "VGG16_decapitated")
	vgg.load_weights(base_weights)


	output_model = Model(inputs=img_input, outputs = predictions, name = "VGG16_bottom2_layers_custom_FC_average")
	return output_model


def VGG16_original(input_shape, base_weights = None, include_top = True): 
	"""
	VGG16: Standard VGG16 model. Pre-populated this with trained weights to make the model usable for my use case. Return either original VGG16 or decapitated VGG16. 
	Configuration: data_format => channel_last
	Args: 
	input_shape: shape of image, including the channel. 
	base_weights: Path to file with pre-trained weights. 
	include_top: If true, then use original weights for convolutional layers, and add new FCs to the top. 
	Return: 
	An instantiated model.
	"""
	img_input = Input(shape=input_shape)

	# Block 1
	# When using Conv2D as first layer, need to correctly connect the layer to input with input_shape argument. 
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
	f1 = x

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
	f2 = x

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
	f3 = x

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
	f4 = x

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
	f5 = x

	# Fully-connected layers (needed in order to load original VGG16 imagenet weights). 
	x = Flatten(name='flatten')(x)
	x = Dense(4096, activation='relu', name='fc1')(x)
	x = Dense(4096, activation='relu', name='fc2')(x)
	x = Dense(1000, activation='softmax', name='predictions')(x) # Since loading pre-trained weights, need VGG classes. 


	# Returns either original VGG16 or VGG16 model without FCs. 
	# Loads either 1) the entire VGG16 network or 2) the convolutional layers. 
	if (include_top): 
		# Need to instantiate model in order to load weights (and return it)
		vgg  = Model(img_input, x, name = "VGG16_original")

		# Load all weights into model. 
		vgg.load_weights(base_weights)

		# Return the original VGG16 model with original FCs. 
		# Full VGG16 Network: Instantiates a model given the input and output layers. 
		output_model  = vgg
	else:

		# Decapitate the top of the VGG16 model. 
		# If weights are already loaded into layers, other models will use these initialized weights. 
		vgg  = Model(inputs=img_input, outputs = f5, name = "VGG16_decapitated")

		# Load all weights into model. 
		vgg.load_weights(base_weights)

		output_model  = vgg


	return output_model
