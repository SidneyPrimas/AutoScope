#Import basic libraries
import glob
import random
#Import keras libraries
from tensorflow.python.keras.optimizers import SGD, RMSprop, Adadelta, Adam
#Import local libraries
import CNN_functions

""""
Implementation Notes: 
+ target_size and fullscale_target_size need to have symmetrical dimensions. If they are not symmetrical, need to remove my PIL usage (where the shape is defined differently than cv2).
"""

# Configuration for foreground/background segmentation
class ClassifyParticles_Config():
	def __init__(self):
		# Core Configurations: Manually updated by user. Always needed. 
		self.project_folder = "20171230_binary_entireImage/"
		self.root_data_dir =  "./urine_particles/data/CICS_experiment/"
		self.weight_file_input_name = None #Set to 'None' to disable.
		self.weight_file_output_name = "classify_weights_" # Set to 'None' to disable. 
		self.target_size = (52, 52) # (height, wdith)
		self.batch_size = 25 
		self.num_epochs = 5  # Print validation results after each epoch. Save model after num_epochs.
		self.batches_per_epoch_train = 10 # Batches for each training session. If None, set so that every image is trained. 
		self.batches_per_epoch_val = 4 # Batches for each validation session. If None, set so that every image is trained. 
		self.nclasses = 4

		# Secondary Configurations: Manually updated by user. Sometimes needed. 
		self.optimizer = Adam(lr=1e-4) # Adam seems to work much better. 
		self.debug = 1
		self.imagenet_weights_file = "./model_storage/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5" # Always needed! 
		self.save_aug_data_to_dir = False
		self.channels = 3
		

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
		self.log_file_name = "classification2.log" #If None, then name based on datetime.
		self.logger = CNN_functions.create_logger(self.log_dir, self.log_file_name, self.log_name)


		