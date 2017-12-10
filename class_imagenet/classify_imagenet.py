"""
Credit To: https://github.com/DeepLearningSandbox/DeepLearningSandbox/blob/master/image_recognition/classify.py
Implementation Note: 
In this implementation, I used cv2 for image importing and image manipulation. In the future, use PIL since the network is trained on PIL.cv2 uses slightly different configurations: BGR instead of RGB, switches width/height and has different resizing techniques. I have adjusted for these differences in this code. 
"""

# Import basic libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from PIL import Image

#Import keras libraries
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras._impl.keras import backend as K

# Configuration variables
target_size = (224, 224)
base_weights_path = "./model_storage/vgg16_weights_tf_dim_ordering_tf_kernels.h5"

# Import Model
from vgg_model import VGG16 as model
model = model(base_weights = base_weights_path, classes = 1000, transfer_learning = False)
# from tensorflow.python.keras.applications.resnet50 import ResNet50 as model
# model = model(weights='imagenet')



def predict(model, img, target_size, top_n=3):
	"""Run model prediction on image
	Args:
	model: keras model
	img: numpy format image
	target_size: (width, height) tuple
	top_n: # of top predictions to return
	Returns:
	list of predicted labels and their probabilities
	"""

	# General Note: Ensure that image array is in correct format. 
	# Currently, we are setup as channel last, so: (num_images, height, width, dimension) 

	# Ensure that image has shape compatiple with original imagenet configuration.  
	if img.shape != target_size:
		#img = img.resize(target_size) #(PIL implementation)
		img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)


	#x = image.img_to_array(img) #(PIL Implementation)

	# Add dimension to beginning of numpy array that represents different images. Essentially, add batch size. 
	x = np.expand_dims(img, axis=0).astype('float32')
	# Data normalization step that zero centers the image data using mean channel values (across entire imagenet dataset). 
	# Data normalization done to normalize input features. Each channel then centered around a 0-mean. 
	# vgg16, resnet, and vgg19 have same preprocess function. (assumes PIL input)
	x = preprocess_input(x, data_format = 'channels_last')
	# Used to predict image outputs without training or validation. 
	# Returns the final prediction tensor (after softmax) with the probabilites of 1k classes. 
	preds = model.predict(x, verbose=1)
	# Application function from resnet and vgg to decode the class outputs. 
	# Uses imagenet file (https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) to convert from class to human readable object. 
	return decode_predictions(preds, top=top_n)[0] # Returns a list within a list. Remove the upper list since it's just a shell. 

def plot_preds(image, preds):  
	"""Displays image and the top-n predicted probabilities 
	 in a bar graph  
	Args:    
	image: numpy image
	preds: list of predicted labels and their probabilities  
	"""  
	#image
	plt.figure() 
	plt.subplot(211)
	plt.imshow(image)
	plt.axis('off')

	#bar graph 
	plt.subplot(212)
	order = list(reversed(range(len(preds))))  
	# Extract the predictions
	bar_preds = [pr[2] for pr in preds]
	# Extract the labels
	labels = (pr[1] for pr in preds)
	plt.barh(order, bar_preds, alpha=0.5)
	plt.yticks(order, labels)
	plt.xlabel('Probability')
	plt.xlim(0, 1.01)
	plt.tight_layout()
	plt.show()



if __name__=="__main__":
	a = argparse.ArgumentParser()
	a.add_argument("--image", help="path to image")
	args = a.parse_args()

if args.image is None:
	a.print_help()
	sys.exit(1)

# Print Configuration Setup
print "Image Size: %s"%(str(target_size))
print "Image Formatting: %s"%(K.image_data_format())

if args.image is not None:
	print "Image Location: %s"%(args.image)
	#img = Image.open(args.image) #(PIL Implementation)
	#Important: CV2 imports the image as BGR (not RGB). Change configuration. 
	img = cv2.imread(args.image)
	img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	preds = predict(model, img, target_size)
	plot_preds(img, preds)

