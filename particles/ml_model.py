import datetime

# Notes: Class to store relevent paremeters in creating and running tensorflow. 

class ML_Model_Parameters(object):
    """ ML_Model stores important parameters for the Tensorflow Graph """

    def __init__(self):
        """ Initialize key parameters for a Tensorflow Graph """
        self.directory_map = {}
        self.train_batch_size = 0
        self.validation_batch_size = 0
        self.step_display = 0
        self.step_save = 0
        self.learning_rate = 0
        self.dropout = 0
        self.target_dim = 0
        self.class_size = 0
        self.data_directory = ""
        self.filter_path = ""
        self.fc_layers_path = ""


        # Creat Log (Append data, and flush to file)
        now = datetime.datetime.now()
        log_file_name = now.strftime("./log/log_%Y-%m-%d_%H:%M:%S")
        self.log=open(log_file_name,  'a', 1000)