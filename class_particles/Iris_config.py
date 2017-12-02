#Import basic libraries
import glob
#Import keras libraries
from tensorflow.python.keras.optimizers import SGD, RMSprop, Adam
#Import local libraries
import CNN_functions

class IrisConfig_Base:
	# Core Configurations: Manually updated by user. Always needed. 
	root_dir =  "./class_particles/data/image_data/20171202_input_data/"
	custom_weight_file_input = None #"./class_particles/data/model_storage/20171126_experiments/20171126_vgg16_customTF_dev.h5" #Set to none to disable
	tl_model_file = "./class_particles/data/model_storage/20171126_experiments/20171201_vgg16_customTL_dev.h5"
	ft_model_file = "./class_particles/data/model_storage/20171126_experiments/20171201_vgg16_customFT_dev.h5"
	log_dir = "./class_particles/data/log/20171202_irisClassification/"
	target_size = (56, 56)
	batch_size = 64
	num_epochs = 2
	channels = 3
	optimizer = Adam(lr=1e-4) # Adam seems to work much better. 

	# Secondary Configurations: Manually updated by user. Sometimes needed. 
	original_data_dir = "./data_storage/IRIS_data/IrisDB_base/"
	imagenet_weights_file = "./model_storage/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5" # Always needed! 
	validation_percent = 0.2
	tl_freeze_layer = "block3_pool"
	ft_freeze_layer = "block2_pool"
	save_to_dir_bool = False

	# Optional Configurations: Either auto calculated or manually filled in. Print results at end of every epoch. 
	batches_per_epoch_train = 2 # Batches for each training session. If None, set so that every image is trained. 
	batches_per_epoch_val = 1 # Batches for each validation session. If None, set so that every image is trained. 


	# Auto Configurations: Can be auto-calculated. 
	train_dir = root_dir + "Training/"
	val_dir = root_dir + "Validation/"
	image_shape = target_size + (channels,)
	classes = len(glob.glob(train_dir + "*"))

	# Create and configure logger. 
	log_name = "TF_logger"
	log_file_name = None #"experiment.log" #If None, then name based on datetime.
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
