#Import basic libraries
import glob
#Import keras libraries
from tensorflow.python.keras.optimizers import SGD, RMSprop, Adadelta
#Import local libraries
import CNN_functions


class SegmentParticles_Base:
	# Core Configurations: Manually updated by user. Always needed. 
	root_data_dir =  "./segment_particles/data/"
	weight_file_input = root_data_dir + "model_storage/20171205_experiment/20171208_vgg16_customTL_dev_6000_selected.h5"  #Set to 'None' to disable
	weight_file_output = root_data_dir + "model_storage/20171205_experiment/20171208_vgg16_customTL_dev_"
	log_dir = root_data_dir + "log/20171205_experiment/"
	target_size = (128, 128)
	batch_size = 2
	num_epochs = 1
	optimizer = Adadelta() # Previously, used Adam. 
	debug = 1
	output_img_dir = root_data_dir + "image_data/20171205_experiment/img_output/"


	# Secondary Configurations: Manually updated by user. Sometimes needed. 
	train_images_dir = root_data_dir + "image_data/20171205_experiment/train_images/"
	train_annotations_dir = root_data_dir + "image_data/20171205_experiment/train_annotations/"
	val_images_dir = root_data_dir + "image_data/20171205_experiment/val_images/"
	val_annotations_dir = root_data_dir + "image_data/20171205_experiment/val_annotations/"
	imagenet_weights_file = "./model_storage/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5" # Always needed! 
	save_aug_data_to_dir = False
	nclasses = 2
	channels = 3

	# Optional Configurations: Either auto calculated or manually filled in. Print results at end of every epoch. 
	batches_per_epoch_train = 2 # Batches for each training session. If None, set so that every image is trained. 
	batches_per_epoch_val = 1 # Batches for each validation session. If None, set so that every image is trained. 


	# Auto Configurations: Can be auto-calculated. 
	image_shape = target_size + (channels,)

	# Create and configure logger. 
	log_name = "TF_logger"
	log_file_name = "experiment.log" #If None, then name based on datetime.
	logger = CNN_functions.create_logger(log_dir, log_file_name, log_name)
	
