"""
    File name: confusion_matrix.py
    Author: Sidney Primas
    Date created: 11/15/2017
    Python Version: 2.7
    Description: Visualize confusion matrix. 
"""


import numpy as np
import matplotlib.pyplot as plt
import sys
import confusion_utility

# Import homebrew functions
base_directory = "./segment_particles/data/CICS_experiment/log/end_to_end_test/"
log_name = "classification1.log"
sys.path.append(base_directory)
import config 

log_path = base_directory + log_name
total_input = 20

cnf_matrix = confusion_utility.get_confusion_matrix(log_path, total_input, config.logs['num_of_classes'])


### Plot Results ###
# Plot non-normalized confusion matrix
confusion_utility.plot_confusion_matrix(cnf_matrix, classes=config.logs['class_names'], title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
confusion_utility.plot_confusion_matrix(cnf_matrix, classes=config.logs['class_names'], normalize=True, title='Normalized confusion matrix')

# Print the classification stats. 
confusion_utility.display_classification_stats(cnf_matrix, config.logs['class_names'])

plt.show()