"""
    File name: model_factory_particles.py
    Author: Sidney Primas
    Date created: 12/03/2017
    Python Version: 2.7
    Description: Includes a factory of keras models. 
"""
# Import basic libraries
import tensorflow as tf
import numpy as np
import h5py  

# Import keras libraries
from tensorflow.python.keras.models import Model # Allows to build more complex models than Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Input, Dropout, Conv2DTranspose, Cropping2D, Add, Reshape, Permute, Activation # Import custom layers. 
from tensorflow.python.keras._impl.keras import backend as K


def FCN8_32px_factor(input_shape, base_weights, classes):
	"""
	FCN8: FCN8 semantic segmentation model as implemented in https://arxiv.org/abs/1411.4038. 
	Configuration: data_format => channel_last
	Args: 
	input_shape: shape of image, including the channel. The format is (input_height, input_width ,channels)
	base_weights: Path to file with pre-trained weights. 
	Return: 
	An instantiated model.
	Notes: Comments explain model/data with a 64x64px input. Model design for input image size that is a fact of 32px. 
	Reference: Implementation inspired from https://github.com/divamgupta/image-segmentation-keras. 
	"""	

	# Verify assumptions 
	# The results from TransposeConvolutions are variable based on input image size. This leads to different types of crops.
	# Since cropping implementation is based on input image, this network has been designed for 64x64 images. 
	assert input_shape[0]%32 ==  0
	assert input_shape[1]%32 == 0
	assert K.image_data_format() == 'channels_last'


	img_input = Input(shape=input_shape)

	### Econding Convolutional Layers (from VGG16) ###
	# Block 1
	# Description: For the data, the format is (rows, cols, channels). For the filters, (kernel size, kernel size, input channels, output channels)
	# Note: Biases used in all 2D convolutions. 
	# Input: (64, 64, 3), Output: (64, 64, 64), Filter: (3,3,3,64)
	# Description: The filter transforms a 3 channel image into a 64 channel image. 
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
	# Input: (64, 64, 64), Output: (64, 64, 64), Filter: (3,3,64,64)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
	# Input: (64, 64, 64), Output: (32, 32, 64)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
	f1 = x

	# Block 2
	# Input: (32, 32, 64), Output: (32, 32, 128), Filter: (3,3,64,128)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
	# Input: (32, 32, 128), Output: (32, 32, 128), Filter: (3,3,128,128)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
	# Input: (32, 32, 128), Output: (16, 16, 128)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool',)(x)
	f2 = x

	# Block 3
	# Input: (16, 16, 128), Output: (16, 16, 256), Filter: (3,3,128,256)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	# Input: (16, 16, 256), Output: (16, 16, 256), Filter: (3,3,256,256)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	# Input: (16, 16, 256), Output: (16, 16, 256), Filter: (3,3,256,256)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
	# Input: (16, 16, 256), Output: (8, 8, 256)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
	f3 = x

	# Block 4
	# Input: (8, 8, 256), Output: (8, 8, 512), Filter: (3,3,256,512)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
	# Input: (8, 8, 512), Output: (8, 8, 512), Filter: (3,3,512,512)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
	# Input: (8, 8, 512), Output: (8, 8, 512), Filter: (3,3,512,512)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
	# Input: (8, 8, 512), Output: (4, 4, 512)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
	f4 = x

	# Block 5
	# Input: (4, 4, 512), Output: (4, 4, 512), Filter: (3,3,512,512)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
	# Input: (4, 4, 512), Output: (4, 4, 512), Filter: (3,3,512,512)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
	# Input: (4, 4, 512), Output: (4, 4, 512), Filter: (3,3,512,512)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
	# Input: (4, 4, 512), Output: (2, 2, 512)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
	f5 = x

	# Load VGG16 weights for convolutional layers obtained from imagenet training
	vgg_noFC  = Model(  img_input , f5  )
	vgg_noFC.load_weights(base_weights)


	### Converting fully connected layers (fc6 and fc7) to fully convolutional layers. ####
	o = f5
	# Input: (2, 2, 512), Output: (2, 2, 1024), Filter: (7,7,512,1024)
	# Description: The filter transforms a 512 channel image to a 4096 channel image. We have 5096 (7x7x512) filters. 
	o = ( Conv2D( 1024 , ( 7 , 7 ) , activation='relu' , padding='same'))(o)
	o = Dropout(0.5)(o)
	# Input: (2, 2, 1024), Output: (2, 2, 1024), Filter: (1,1,1024,1024)
	# Description: Used to add a non-linearity and mimics a fully connected layer. 
	o = ( Conv2D( 1024 , ( 1 , 1 ) , activation='relu' , padding='same'))(o)
	o = Dropout(0.5)(o)
	# Description: This final convolutional layers produces a heatmap, indicating the presence of a class at each pixel. That means we produce a class at each of the 4 pixels. 
	# Input: (2, 2, 1024), Output: (2, 2, classes), Filter: (1,1,1024,classes)
	o = ( Conv2D( classes ,  ( 1 , 1 ) ,kernel_initializer='he_normal', name='pixel_predictions_base'))(o)

	### Decoding: Upsample the pixel-wise semantic segmentation  ###
	# Decode 1
	# Description: To implement a transpose convolution, we put zeros bilinearly into the input (insert zeros between the values), and then zero pad the input on the edges. The idea is to get the inverse of a nomral 2d convolution with max-pooling. 
	# Note: The 6x6 output is based on both kernel size and input image size. 
	# Note: Transpose convolution sometimes called deconvolution. 
	# Note: 'Valid' padding means that there i no padding. For a transpose convolution, this means that we have to zero pad the input (think about a how normal convolution without padding leads to a result with extra fields. Now, inverse this process for the transpose.)
	# Input: (2, 2, classes), Output: (6, 6, classes), Filter: (4,4,classes,classes)
	upsample1_2x = Conv2DTranspose( classes , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, padding='valid', name='upsample1_2x')(o)

	# Description: Crop the upsampled image upsample1_2x to be the same size as the output features from block 4, skip_block4
	# Note: Cropping size calculation done manualy, and not automated. 
	# Input: (6, 6, classes), Output: (4, 4, classes)
	upsample1_2x_cropped = Cropping2D(cropping = ((1, 1), (1, 1)), name='upsample1_2x_cropped')(upsample1_2x)

	# Descroption: Converts the feature layer to a class. Each pixel gets a class based on all the features corresponding ot that pixel. 
	# Input: (4, 4, 512), Output: (4, 4, classes) Filter: (1, 1, 512, classes)
	skip_block4 = ( Conv2D( classes ,  ( 1 , 1 ) ,kernel_initializer='he_normal', name = 'skip_block4'))(f4)
	
	# Description: Fuse upsampled 2x image with blcok4 output features. 
	# Input: (4, 4, classes) and (4, 4, classes), Output: (4, 4, classes)
	fuse1_out = Add(name='fuse1_out')([ upsample1_2x_cropped , skip_block4 ])

	# Decode 2
	# Input: (4, 4, classes), Output: (10, 10, classes), Filter: (4,4,classes,classes)
	upsample2_2x = Conv2DTranspose( classes , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, name='upsample2_2x')(fuse1_out)
	# Input: (10, 10, classes), Output: (8, 8, classes)
	upsample2_2x_cropped = Cropping2D(cropping = ((1, 1), (1, 1)), name='upsample2_2x_cropped')(upsample2_2x)
	# Input: (8, 8, 256), Output: (8,8, classes), Filter: (1,1,256,classes)
	skip_block3 = ( Conv2D( classes ,  ( 1 , 1 ) ,kernel_initializer='he_normal', name='skip_block3'))(f3)
	# Input: (8, 8, classes) and (8,8,classes), Output: (8,8, classes)
	fuse2_out  = Add(name='fuse2_out')([ skip_block3 , upsample2_2x_cropped ])

	# Decode 3
	# Description: Upsamples the image by 8x.  
	# Input: (8, 8, classes), Output: (72, 72, classes), Filter: (16,16,classes,classes)
	upsample3_8x = Conv2DTranspose( classes , kernel_size=(16,16) ,  strides=(8,8) , use_bias=False, name='upsample3_8x')(fuse2_out)
	# Input: (72, 72, classes), Output: (64, 64, classes)
	upsample3_8x_cropped = Cropping2D(cropping = ((4, 4), (4, 4)), name='upsample3_8x_cropped')(upsample3_8x)
	
	o_shape = Model(img_input , upsample3_8x_cropped ).output_shape
	outputHeight = o_shape[1]
	outputWidth = o_shape[2]
	# Note: If value is -1, value is inferred from remaining dimensions. 
	# Input: (64, 64, classes), Output: (5184, classes)
	reshape = (Reshape((outputHeight*outputWidth, -1  ), name='reshape'))(upsample3_8x_cropped)
	# Note: By default, softmax is applied to last dimension. 
	output = (Activation('softmax'))(reshape)
	model = Model( img_input , output )
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight

	# Verify assumptions
	assert model.outputWidth == input_shape[0]
	assert model.outputHeight == input_shape[1]


	return model