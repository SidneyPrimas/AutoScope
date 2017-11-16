log_directory =  "./log/20170419_originalDownsampling/"


# Selected
logs = dict(
	url = [], 
	file_name = ["log_52px_0p54", "log_52px_0p7", "log_52px_1p4", "log_52px_2p5", "log_52px_4p0", "log_52px_8p0", "log_52px_10p0", "log_52px_12p0", "log_52px_16p0"],
	legend = [0.54, 0.7, 1.4, 2.5, 4.0, 8.0, 10.0, 12.0, 16.0],  
	class_names = ["Bact", "RBC", "WBC", "Other"], 
	num_of_classes = 4, 
	resolution = []
	)

# # Files with *VARIABLE* image size and 4 classes
# file_name = ["log_0p54", "log_0p7", "log_1p2", "log_1p4", "log_1p7", "log_2p0", "log_2p5", "log_4p0"]
# legend = [0.54, 0.7, 1.2, 1.4, 1.7, 2.0, 2.5, 4.0]
# class_names = ["Bact", "RBC", "WBC", "Other"]

# # Files with *FIXED* sizes at 4 classes. 
# file_name = ["log_52px_0p54", "log_52px_0p7", "log_52px_1p4", "log_52px_2p5", "log_52px_4p0", "log_52px_8p0", "log_52px_10p0", "log_52px_12p0", "log_52px_16p0"]
# legend = [0.54, 0.7, 1.4, 2.5, 4.0, 8.0, 10.0, 12.0, 16.0]
# class_names = ["Bact", "RBC", "WBC", "Other"]


# # Files with *FIXED* sizes at 5 classes. 
# # Important Note: Class categorization is incorrect!
# file_name = ["log_52px_0p54_5C", "log_52px_4p0_5C"]
# legend = [0.54, 4.0]
# class_names = ["Bact", "RBC", "WBC", "Some Cast + Some NSEP", "Other"]