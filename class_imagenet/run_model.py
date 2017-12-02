# Import basic libraries
import tensorflow as tf
import numpy as np
import cv2

#Import keras libraries
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.optimizers import SGD

# Import local libraries
import vgg_model_imagenet as model


VGG16_base_weights_path = "./model_storage/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
model = model.VGG16_original(input_shape = (224, 224, 3) , base_weights = VGG16_base_weights_path)
plot_model( model , show_shapes=True , to_file='model.png')

 
im = cv2.imread('./data/random_datasets/cat_dog_data/pred_dir/95.jpg')
im = cv2.resize(im, (224, 224)).astype(np.float32)

im[:,:,0] -= 103.939
im[:,:,1] -= 116.779
im[:,:,2] -= 123.68

im = np.expand_dims(im, axis=0)

# Test pretrained model
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
out = model.predict(im)
print np.argmax(out)

