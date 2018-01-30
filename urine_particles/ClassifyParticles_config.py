#Import basic libraries
import glob
import random
#Import keras libraries
from tensorflow.python.keras.optimizers import SGD, RMSprop, Adadelta, Adam
#Import local libraries
import CNN_functions

""""
Execution Notes: 
+ RGB vs. Grayscale: 
++ To switch between RGB and grayscale: 1) adjust self.channels, 2) adjust self.color, 3) adjust normalization in preprocessing function.

Implementation Notes: 
+ Batch Size: 
++ Due to the Keras implementation of flow_from_directory, the returned batch size varies at times. Keep this in mind when analyzing results.
+ Image dimensions: 
++ TF assumes (height, width) format. However, Keras loads images with PIL, so it assumes (widht, height) format. Be careful with reconciling. 

"""

# Configuration for foreground/background segmentation
class ClassifyParticles_Config():
	def __init__(self):
		# Core Configurations: Manually updated by user. Always needed. 
		self.project_folder = "20180120_training/"
		self.root_data_dir =  "./urine_particles/data/clinical_experiment/"
		self.weight_file_input_name =  "classification_base_model_ImageNorm_grayscale.h5" #Set to 'None' to disable.
		self.weight_file_output_name = "classify_weights_" # Set to 'None' to disable. 
		self.target_size = (64, 64) # Warning: Be careful if non-square dimensions (see above note). 
		self.batch_size = 64 
		self.num_epochs = 5  # Print validation results after each epoch. Save model after num_epochs.
		self.batches_per_epoch_train = 35 # Batches for each training session. If None, set so that every image is trained. 
		self.batches_per_epoch_val = 5 # Batches for each validation session. If None, set so that every image is trained. 
		self.nclasses = 4

		# Secondary Configurations: Manually updated by user. Sometimes needed. 
		self.optimizer = Adam(lr=1e-4) # Adam seems to work much better. 
		self.debug = 1
		self.imagenet_weights_file = "./model_storage/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5" # Always needed! 
		self.save_aug_data_to_dir = False
		self.channels = 1
		self.color = 'grayscale' # Select 'rgb' or 'grayscale'. Remember to adjust normalization script in preprocessing function. 
		self.preprocess_func = "gray_imageNorm" # Options include => "gray_imageNorm", "rgb_imageNorm", "rgb_datasetNorm"

		# Auto Configurations: Can be auto-calculated. 
		self.train_images_dir = self.root_data_dir + "image_data/" + self.project_folder + "classification/training/"
		self.val_images_dir = self.root_data_dir + "image_data/" + self.project_folder + "classification/validation/"
		self.weight_file_output = None if self.weight_file_output_name is None else (self.root_data_dir + "model_storage/" + self.project_folder + self.weight_file_output_name)
		self.log_dir = self.root_data_dir + "log/" + self.project_folder
		self.output_img_dir = self.root_data_dir + "image_data/" + self.project_folder +"classification/debug_output/"
		self.weight_file_input = None if self.weight_file_input_name is None else (self.root_data_dir + "model_storage/" + self.project_folder + self.weight_file_input_name)  
		self.image_shape = self.target_size + (self.channels,)
		self.classification_metadata = CNN_functions.get_json_log(self.root_data_dir + "image_data/" + self.project_folder + "classification_metadata.log")
		

		# Create and configure logger. 
		self.log_name = "TF_logger"
		self.log_file_name = "classification.log" #If None, then name based on datetime.
		self.logger = CNN_functions.create_logger(self.log_dir, self.log_file_name, self.log_name)


		
