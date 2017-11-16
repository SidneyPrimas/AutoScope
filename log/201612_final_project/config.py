log_directory =  "./log/201612_final_project/"


# Selected: Currently only meant to run comparsion across logs
logs = dict(
	url = [], 
	file_name = ["log_4classesOther_basicGraph_imagesPerClass_newImages_111", "log_6classesOther_basicGraph_imagesPerClass_newImages_163", "log_7classes_other_basicGraph_imagesPerClass_newImages_179"],
	legend = ["4 Classes", "6 Classes", "7 Classes" ],
	num_of_classes = 0,
	class_names = [],
	resolution = []
	)

# # Comparing different classes. 
# file_name = ["log_4classesOther_basicGraph_imagesPerClass_newImages_111", "log_6classesOther_basicGraph_imagesPerClass_newImages_163", "log_7classes_other_basicGraph_imagesPerClass_newImages_179"],
# legend = ["4 Classes", "6 Classes", "7 Classes" ],

# # Comparing different architectures
# file_name = ["log_6classesOther_basicGraph_imagesPerClass_newImages_163", "log_6lcassesOther_2by2C-Layers_imagesPerClass_newImages_122"], 
# legend = ["Single 5x5px Kernel", "Two 3x3px Kernels"], 
# num_of_classes = 6,
# class_names = ["Bact", "RBC", "Sperm", "WBC", "Cryst", "Other"],


# # Agumented Data Across Different Classes
# file_name = ["log_4classes_basicGraph_imagesPerClass_augmented_193", "log_6classes_basicGraph_imagesPerClass_augmented_178", "log_7classes_basicGraph_imagesPerClass_augmented_185", "log_6lcasses_2by2C-Layers_imagesPerClass_augmented_154"],
# legend = ["4 Classes (Base Graph) ", "6 Classes (Base Graph)",  "7 Classes (Base Graph)", "6 Classes (Two 3x3 Kernels)"],



# # Training variations
# file_name = ["log_6classesOther_basicGraph_imagesPerClass_newImages_163", "log_6classesOther_basicGraph_Random_newImages_withWeights_178", "log_6classesOther_basicGraph_Random_newImages_noWeights_174"],
# legend = ["Equal Images Per Class", "Randomly Selected (weighted)", "Randomly Selected (not weighted)"],
# num_of_classes = 6,
# class_names = ["Bact", "RBC", "Sperm", "WBC", "Cryst", "Other"],