#Import basic libraries
import glob
#Import keras libraries
from tensorflow.python.keras.optimizers import SGD, RMSprop, Adam
#Import local libraries
import CNN_functions

class IrisConfig_Base:
	# Core Configurations: Manually updated by user. Always needed. 
	root_dir =  "./data/IRIS_data/keras_directory/20171127_input_data/"
	input_weights_file = "./model_storage/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
	output_model_file = "./model_storage/v2_classification/IRIS/20171126_vgg16_customTF_dev.h5"
	log_dir = "./log/keras/20171127_iris_experiment/"
	target_size = (56, 56)
	batch_size = 64
	epoch = 1
	channels = 3
	optimizer = Adam(lr=1e-4) # Adam seems to work much better. 

	# Secondary Configurations: Manually updated by user. Sometimes needed. 
	original_data_dir = "./data/IRIS_data/IrisDB_base/"
	validation_percent = 0.2
	tl_freeze_layer = "block5_pool"
	save_to_dir_bool = False

	# Optional Configurations: Either auto calculated or manually filled in. 
	batches_per_epoch_train = 1 # Batches for each training session. If None, set so that every image is trained. 
	batches_per_epoch_val = 1 # Batches for each validation session. If None, set so that every image is trained. 


	# Auto Configurations: Can be auto-calculated. 
	train_dir = root_dir + "Training/"
	val_dir = root_dir + "Validation/"
	image_shape = target_size + (channels,)
	classes = len(glob.glob(train_dir + "*"))

	# Create and configure logger. 
	log_name = "TF_logger"
	log_file_name = "experiment.log" #If None, then name based on datetime.
	logger = CNN_functions.create_logger(log_dir, log_file_name, log_name)
	

	# Maps particle folder names to class names.
	directory_map = {
		"IRIS-BACT": "Bacteria",
		"IRIS-RBC": "RBC",
		"IRIS-SPRM": "Other",
		"IRIS-WBC": "WBC",
		"IRIS-CLUMP-WBCC": "WBC",
		"IRIS-CLUMP-YSTS": "Other",
		"IRIS-CRYST-CAOX": "Other",
		"IRIS-CRYST-CAPH": "Other",
		"IRIS-CRYST-TPO4": "Other",
		"IRIS-CRYST-URIC": "Other",
		"IRIS-HYAL": "Hyaline_Casts",
		"IRIS-NHYAL-CELL": "Other",
		"IRIS-NHYAL-GRAN": "Other",
		"IRIS-NSQEP-REEP": "Other",
		"IRIS-NSQEP-TREP": "Other",
		"IRIS-SQEP": "SQEP",
	}
