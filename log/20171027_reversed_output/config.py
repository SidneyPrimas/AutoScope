log_directory =  "./log/20171027_reversed_output/"

logs = dict(
	url = ["ubuntu@ec2-18-221-87-180.us-east-2.compute.amazonaws.com", "ubuntu@ec2-13-58-141-119.us-east-2.compute.amazonaws.com"], 
	file_name = ["20171027_irl_wbc_rbc_10um_v1", "20171027_irl_wbc_rbc_10um_v2"], 
	legend = ["V1", "V2"], 
	class_names = ["RBC", "WBC", "10um", "Other"], 
	num_of_classes = 4, 
	resolution = [1.36, 1.36]
	)