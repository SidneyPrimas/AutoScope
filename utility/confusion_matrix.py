import numpy as np
import matplotlib.pyplot as plt
import sys
import confusion_utility

"""
Description: Visualize confusion matrix. 
"""

# Import homebrew functions
base_directory = "./urine_particles/data/clinical_experiment/log/20180205_training_plus/classification_training_log/"
log_name = "20180211_final_model_highAug_v2.log"
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