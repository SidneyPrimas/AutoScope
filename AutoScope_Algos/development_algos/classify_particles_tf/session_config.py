import datetime
import numpy as np

""" 
Description: SessionConfiguration_Iris stores important parameters for the Tensorflow Graph and Session 
"""

class SessionConfiguration_Iris(object):

	def __init__(self):

		# Manually update classification mapping
		self.directory_map = {
			"IRIS-BACT": 0,
			"IRIS-RBC": 1,
			"IRIS-SPRM": 5,
			"IRIS-WBC": 2,
			"IRIS-CLUMP-WBCC": 2,
			"IRIS-CLUMP-YSTS": 5,
			"IRIS-CRYST-CAOX": 5,
			"IRIS-CRYST-CAPH": 5,
			"IRIS-CRYST-TPO4": 5,
			"IRIS-CRYST-URIC": 5,
			"IRIS-HYAL": 3,
			"IRIS-NHYAL-CELL": 5,
			"IRIS-NHYAL-GRAN": 5,
			"IRIS-NSQEP-REEP": 5,
			"IRIS-NSQEP-TREP": 5,
			"IRIS-SQEP": 4,
		}


		# Manually updated configuration parameters
		self.train_batch_size = 120
		self.validation_batch_size = 120
		self.num_epochs = 10
		self.batches_per_epoch_train = 40
		self.batches_per_epoch_val = 2
		self.learning_rate = 1e-4
		self.dropout = 0.5
		self.target_dim = 116
		self.channels = 1
		self.class_size = len(np.unique(self.directory_map.values()))
		self.data_directory = 	"./data_storage/IrisDB_notResampled/"
		self.log_dir_path = 	"./classify_particles_tf/data/log/20171231_refactor_code/"

		# Filter parameters
		# Note: Set to None to block restoring or saving graph variables. 
		# Note: Goldstandard filters => #"./classify_particles_tf/data/model_storage/v1_original_TF_models/201612_class_models/download_in_201612/particle_model_filters_2times2Layers_from_154_v1"
		self.filter_load_path = 	None #"./classify_particles_tf/data/model_storage/20171231_refactor_code/particle_model_filters_2times2Layers" 
		self.fc_layers_load_path = 	None #"./classify_particles_tf/data/model_storage/20171231_refactor_code/particle_model_fc_layers-4"
		self.filter_save_path = 	"./classify_particles_tf/data/model_storage/20171231_refactor_code/particle_model_filters_2times2Layers"
		self.fc_layers_save_path = 	"./classify_particles_tf/data/model_storage/20171231_refactor_code/particle_model_fc_layers-%d"%(self.class_size)

		# Configuration parameters
		self.equal_images_per_class = True
		self.visualize_filters = False
		self.debug = False
		self.resize_images = True


		# State parametes
		self.image_trained_count = 0

		# Auto-created parameters
		self.weights = None


		# Creat Log (Append data, and flush to file)
		now = datetime.datetime.now()
		log_file_name = now.strftime("log_%Yy%mm%dd_%Hh%Mmin%Ssec")
		log_path = self.log_dir_path + log_file_name
		self.log=open(log_path,  'a', 1)
