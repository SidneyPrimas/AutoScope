#Import basic libraries
import glob
import random
#Import keras libraries
from tensorflow.python.keras.optimizers import SGD, RMSprop, Adadelta
#Import local libraries
import CNN_functions

# Configuration for foreground/background segmentation
class SegmentParticles_Config():
	def __init__(self):
		# Core Configurations: Manually updated by user. Always needed. 
		self.data_project_folder = "20171218_classes_entireImage/"
		self.generate_images_with_cropping = False
		self.root_data_dir =  "./segment_particles/data/"
		self.weight_file_input_name = "20171208_vgg16_customTL_dev_45000_selected.h5" #Set to 'None' to disable.
		self.weight_file_output_name = "20171208_vgg16_customTL_dev_" # Set to 'None' to disable. 
		self.fullscale_target_size = (640, 640)
		self.target_size = (640, 640) # (height, wdith) => (128, 128) for  crops or (640, 640) for entire image
		self.batch_size = 3
		self.num_epochs = 25
		self.optimizer = Adadelta() # Previously, used Adam. 
		self.nclasses = 5
		self.debug = 1


		# Secondary Configurations: Manually updated by user. Sometimes needed. 
		self.imagenet_weights_file = "./model_storage/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5" # Always needed! 
		self.save_aug_data_to_dir = False
		self.channels = 3
		self.input_folders = ["10um", "wbc", "rbc"]
		self.detection_radius = 10

		# Optional Configurations: Either auto calculated or manually filled in. Print results at end of every epoch. 
		self.batches_per_epoch_train = 1 # Batches for each training session. If None, set so that every image is trained. 
		self.batches_per_epoch_val = 1 # Batches for each validation session. If None, set so that every image is trained. 


		# Auto Configurations: Can be auto-calculated. 
		self.train_images_dir = self.root_data_dir + "image_data/" + self.data_project_folder + "train_images/"
		self.train_annotations_dir = self.root_data_dir + "image_data/" + self.data_project_folder + "train_annotations/"
		self.val_images_dir = self.root_data_dir + "image_data/" + self.data_project_folder + "val_images/"
		self.val_annotations_dir = self.root_data_dir + "image_data/" + self.data_project_folder + "val_annotations/"
		self.weight_file_output = None if self.weight_file_output_name is None else (self.root_data_dir + "model_storage/" + self.data_project_folder + self.weight_file_output_name)
		self.log_dir = self.root_data_dir + "log/" + self.data_project_folder
		self.output_img_dir = self.root_data_dir + "image_data/" + self.data_project_folder +"img_output/"
		self.weight_file_input = None if self.weight_file_input_name is None else (self.root_data_dir + "model_storage/" + self.data_project_folder + self.weight_file_input_name)  
		self.image_shape = self.target_size + (self.channels,)
		self.colors = [(0,0,0)] + [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(self.nclasses-1)]
		self.data_config = CNN_functions.get_data_config(self.root_data_dir + "image_data/" + self.data_project_folder + "data_config.log")
		self.labels = self.data_config['class_labels']

		

		# Create and configure logger. 
		self.log_name = "TF_logger"
		self.log_file_name = "experiment_foreground.log" #If None, then name based on datetime.
		self.logger = CNN_functions.create_logger(self.log_dir, self.log_file_name, self.log_name)


		
