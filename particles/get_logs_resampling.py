"""
    File name: get_logs_resampling.py
    Author: Sidney Primas
    Date created: 4/17/2017
    Python Version: 2.7
    Description: Used for the Resolution Threshold Experiment. Update the aws_servers data structure to download logs from an AWS server. 
    The assumption: Any files in the log folder will be downloaded and renamed based on the 2nd column in the aws_servers struct. 
"""

import os

### Execution Directions ###
# Make sure you call code from root directory (all references based from there)
# Update destination folder of logs as needed

### Setting up parameters ###
# Storage structure: [EC2_address, destination_file_name, resolution]
aws_servers = [
	# ["ubuntu@ec2-52-14-207-70.us-east-2.compute.amazonaws.com", 	"log_0p54", 		0.54],
	# ["ubuntu@ec2-52-14-255-209.us-east-2.compute.amazonaws.com", 	"log_0p7", 			0.7],
	# ["ubuntu@ec2-52-14-249-89.us-east-2.compute.amazonaws.com", 	"log_1p2", 			1.2],
	# ["ubuntu@ec2-52-15-187-153.us-east-2.compute.amazonaws.com", 	"log_1p4", 			1.4], 
	# ["ubuntu@ec2-13-58-42-204.us-east-2.compute.amazonaws.com",		"log_1p7",			1.7],
	# ["ubuntu@ec2-52-15-116-80.us-east-2.compute.amazonaws.com", 	"log_2p0", 			2.0], 
	# ["ubuntu@ec2-13-58-115-146.us-east-2.compute.amazonaws.com", 	"log_2p5", 			2.5], 
	# ["ubuntu@ec2-52-14-62-195.us-east-2.compute.amazonaws.com", 	"log_4p0", 			4], 
	# ["ubuntu@ec2-52-14-216-165.us-east-2.compute.amazonaws.com", 	"log_52x_0p7",		0.7],
	# ["ubuntu@ec2-13-58-5-196.us-east-2.compute.amazonaws.com",		"log_52px_1p4", 	1.4], 
	# ["ubuntu@ec2-13-58-130-173.us-east-2.compute.amazonaws.com", 	"log_52px_2p5", 	2.5], 
	# ["ubuntu@ec2-52-15-156-226.us-east-2.compute.amazonaws.com", 	"log_52px_0p54", 		0.54], 
	# ["ubuntu@ec2-52-15-183-108.us-east-2.compute.amazonaws.com", 	"log_52px_4p0", 		4.0], 
	# ["ubuntu@ec2-52-14-255-109.us-east-2.compute.amazonaws.com", 	"log_52px_8p0", 		8.0], 
	# ["ubuntu@ec2-13-58-9-117.us-east-2.compute.amazonaws.com", 		"log_52px_10p0", 		10.0], 
	# ["ubuntu@ec2-13-58-78-86.us-east-2.compute.amazonaws.com", 		"log_52px_12p0", 		12.0], 
	# ["ubuntu@ec2-13-58-74-4.us-east-2.compute.amazonaws.com", 		"log_52px_16p0", 		16.0], 
	["ubuntu@ec2-52-14-144-27.us-east-2.compute.amazonaws.com", 	"log_52px_0p54_5C", 	0.54], 
	["ubuntu@ec2-13-58-11-157.us-east-2.compute.amazonaws.com", 	"log_52px_4p0_5C", 		4.0]	
]

command = "scp -i "
security_key = "./security/sprimas_admin_key-pari_us-east-2.pem"
source_file = "particle_recognition/log/*"
destination_folder = "./log/downsampling_experiment/"

### Get logs from AWS servers ###
for server in aws_servers:
	EC2_address = server[0]
	dest_file_name = server[1]

	os.system(command + security_key + " " + EC2_address + ":" + source_file + " " + destination_folder + dest_file_name)
