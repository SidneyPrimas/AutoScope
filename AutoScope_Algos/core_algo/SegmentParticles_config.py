#Import basic libraries
import glob
import random
#Import keras libraries
from tensorflow.python.keras.optimizers import SGD, RMSprop, Adadelta
#Import local libraries
import CNN_functions

""""
Execution Notes: 
+ Directions for real-time cropping: 
++ To enable real-time cropping, set generate_images_with_cropping to True. 
++ The fullscale_target_size is only used to change the proportions of the original image (which is sometimes needed so that images are resampled in the same way as during training). 
++ By setting fullscale_target_size to none, the image dimensions are not changed at all, leading to no resampling. 
++ When real-time cropping is enabled, target-size sets the crop size, not the resampling size. This is important! 
+ Target Size: 
++ When not doing real-time cropping, the target_size determine the dimensions that the input image will be resized to. 
++ target_size needs to be evenly divided by 32 in order to work with the model.
+ RAM Constraint
++ During execution (if running locally), the system can get RAM constrained. If that's the case, shut down other running programs, especially Chrome. 


Implementation Notes: 
"""

# Configuration for foreground/background segmentation
class SegmentParticles_Config():
	def __init__(self):
		# Core Configurations: Manually updated by user. Always needed. 
		self.project_folder =  "20180205_training_plus/" #"20180130_semantic_seg/"
		self.root_data_dir =  "./urine_particles/data/clinical_experiment/"
		self.weight_file_input_name = "20180125_base_segmentation.h5" # "seg_class_weights_0.h5"   #Set to 'None' to disable.
		self.weight_file_output_name = "seg_class_weights_" # Set to 'None' to disable. 
		self.generate_images_with_cropping = False
		self.fullscale_target_size = None # (height, width) for numpy
		self.target_size = (2464,3264) # (480, 480) for  crops or (2464,3264) for entire image at 8MPX
		self.batch_size = 16
		self.num_epochs = 1 # Print validation results after each epoch. Save model after num_epochs.
		self.batches_per_epoch_train = 40 # Batches for each training session.
		self.batches_per_epoch_val = 1 # Batches for each validation session. 
		self.nclasses = 2


		# Secondary Configurations: Manually updated by user. Sometimes needed. 
		self.optimizer = Adadelta() # Previously, used Adam. 
		self.debug = 1
		self.imagenet_weights_file = "./model_storage/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5" # Always needed! 
		self.channels = 3
		self.input_folders = ["10um", "wbc", "rbc"]
		self.detection_radius = 30 # The proximity that a predicted particle needs to be to a ground truth particle for it to be detected/classified. (needs to be adjusted based on resampling of image)
		self.bgr_means = [90.61598179, 129.97525112, 103.00621832] # VGG is [103.939, 116.779, 123.68]  

		# Auto Configurations: Can be auto-calculated. 
		self.train_images_dir = self.root_data_dir + "image_data/" + self.project_folder + "segmentation/train_images/"
		self.train_annotations_dir = self.root_data_dir + "image_data/" + self.project_folder + "segmentation/train_annotations/"
		self.val_images_dir = self.root_data_dir + "image_data/" + self.project_folder + "segmentation/val_images/"
		self.val_annotations_dir = self.root_data_dir + "image_data/" + self.project_folder + "segmentation/val_annotations/"
		self.weight_file_output = None if self.weight_file_output_name is None else (self.root_data_dir + "model_storage/" + self.project_folder + self.weight_file_output_name)
		self.log_dir = self.root_data_dir + "log/" + self.project_folder
		self.output_img_dir = self.root_data_dir + "image_data/" + self.project_folder +"segmentation/img_output/"
		self.weight_file_input = None if self.weight_file_input_name is None else (self.root_data_dir + "model_storage/" + self.project_folder + self.weight_file_input_name)  
		self.image_shape = self.target_size + (self.channels,)
		#self.colors = [(0,0,0)] + [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(self.nclasses-1)]
		self.colors = [(0,0,0), (0,0,255), (255,0,0),(0,255,0)] # in bgr format
		self.segmentation_metadata = CNN_functions.get_json_log(self.root_data_dir + "image_data/" + self.project_folder + "segmentation_metadata.log")
		self.labels = self.segmentation_metadata['segmentation_labels']

		# Microscope configuration
		self.hpf_area = 0.196349541 # in mm^2
		self.primas_area = 10.1568 # in mm^2
		

		# Create and configure logger. 
		self.log_name = "TF_logger"
		self.log_file_name = "segmentation.log" #If None, then name based on datetime.
		self.logger = CNN_functions.create_logger(self.log_dir, self.log_file_name, self.log_name)


		
