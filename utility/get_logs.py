"""
    File name: get_logs.py
    Author: Sidney Primas
    Date created: 10/26/2017
    Python Version: 2.7
    Description: Use to download and graph log results. Update the aws_servers data structure to download logs from an AWS server. 
    The assumption: Any files in the log folder will be downloaded and renamed based on the 2nd column in the aws_servers struct. 
"""

import os
import sys


# Import homebrew functions
base_directory = "class_particles/data/log/20171202_irisClassification/"
sys.path.append(base_directory)
import config 

### Execution Directions ###
# Execute script from root folder. 
# Update destination folder of logs as needed. 
# Assumption: Assume that the AWS server only has a single log. 

command = "scp -i "
security_key = "./security/sprimas_admin_key-pari_us-east-2.pem"
source_file = "./particle_recognition/" + base_directory +"*" # Assume that the AWS server only has a single log. 


### Get logs from AWS servers ###
for i, server in enumerate(config.logs['url']):
	EC2_address = server
	dest_file_name = config.logs['file_name'][i]

	os.system(command + security_key + " " + EC2_address + ":" + source_file + " " + base_directory + dest_file_name)
