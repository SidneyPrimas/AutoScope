log_directory =  "./log/20170503_micrsocope_and_reversed/"


# Selected
logs = dict(
	url = [], 
	file_name = ["microscope_output/particles_clumps_other2", "reversed_output/particles_clumps_other", "reversed_output/switched_6um_and_10um", "reversed_output/random_validation_files"],
	legend = ["Images from Microscope", "Images from Reversed Lens", "Reversed Lens where Switched 6um and 10um in Validation Dataset", "The Validation Images Are Randomized"],
	class_names = ["6um", "10um", "BAF3", "Other"], 
	num_of_classes = 4, 
	resolution = []
	)

# # Files with *VARIABLE* image size and 4 classes
# file_name = ["microscope_output/particles_clumps_other2", "reversed_output/particles_clumps_other", "reversed_output/switched_6um_and_10um", "reversed_output/random_validation_files"],
# legend = ["Images from Microscope", "Images from Reversed Lens", "Reversed Lens where Switched 6um and 10um in Validation Dataset", "The Validation Images Are Randomized"],
# class_names = ["6um", "10um", "BAF3", "Other"], 
# num_of_classes = 4, 