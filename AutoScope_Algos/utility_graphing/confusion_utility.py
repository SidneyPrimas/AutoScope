import matplotlib.pyplot as plt
import numpy as np
import itertools
import re
from ast import literal_eval


"""
Description: Utility functions used by other print files. 
"""


def get_confusion_matrix(log_path, total_input, class_size):
	"""
	This function gets the confusion matrix from the log file. 
	If indicated by input_number, it averages multiple confusion matrices. 
	"""

	with open(log_path) as f:
		# Keeps track of total inputs
		input_count = 0 
		all_cnf_matrices = []

		inverse_file = f.readlines()
		inverse_file.reverse()
		for i, line in enumerate(inverse_file):
			if (line.find("Confusion Matrix:") >= 0):
				# If file contains partial confusion matrix, continue (eod conditions)
				if ((i-class_size) < 0): continue
				# Extract and format the confusion matrix
				confusion = inverse_file[i-class_size:i]
				confusion.reverse()
				cnf_matrix = convert_to_np_array(confusion)
				all_cnf_matrices.append(cnf_matrix)
				input_count = input_count + 1
				# Once we have the total matrices, break from collection the confusion matrices
				if (input_count >= total_input): 
					print input_count
					break

		cnf_matrix = np.sum(all_cnf_matrices, axis=0) 


	return cnf_matrix

def convert_to_np_array(confustion_list):
	confusion_string = ""

	for i, row in enumerate(confustion_list):
		# Note: The following implementation is inefficient in not elegant. 
		# ToDo: Fix implementation
		row = row.replace('\n', ',').replace('[  ', '[').replace('.','.,').replace(']],',']]')
		# Update the formatted row into confusion_string
		confusion_string = confusion_string + row

	# Literal converts string to literals so np.array doesn't see string. 
	return np.array(literal_eval(confusion_string))




def plot_confusion_matrix(cm, classes, normalize=False, title = '', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if normalize:
        cm = 100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize="16")
    plt.yticks(tick_marks, classes, fontsize="16")
    fig.patch.set_facecolor('white')

    percent_str = ''
    if (normalize):
        percent_str='%'

    fmt = '.0f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt)+percent_str,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Particle Type', fontsize="20")
    plt.xlabel('Predicted Particle Type', fontsize="20")


def display_classification_stats(cm, class_names):
	"""
	This function calculatates and prints different stats 
	related to classification quality. 
	"""

	# Calculate based parameters (with numpy matrix)
	# Use following link as reference: https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
	FP = cm.sum(axis=0) - np.diag(cm)  
	FN = cm.sum(axis=1) - np.diag(cm)
	TP = np.diag(cm)
	TN = cm.sum() - (FP + FN + TP)

	# Calculate and display results. 
	display_data = []
	display_labels = []

	# Sensitivity, hit rate, recall, or true positive rate
	TPR = TP/(TP+FN)
	display_data.append(TPR)
	display_labels.append("Sensitivity (TPR):")
	# Specificity or true negative rate
	TNR = TN/(TN+FP) 
	display_data.append(TNR)
	display_labels.append("Specificity (TNR):")
	# Precision or positive predictive value
	PPV = TP/(TP+FP)
	display_data.append(PPV)
	display_labels.append("Precision (PPV):")
	# Negative predictive value
	NPV = TN/(TN+FN)
	display_data.append(NPV)
	display_labels.append("Negative Predictive Value (NPV):")
	# Fall out or false positive rate
	FPR = FP/(FP+TN)
	display_data.append(FPR)
	display_labels.append("False Positive Rate (FPR):")
	# False negative rate
	FNR = FN/(TP+FN)
	display_data.append(FNR)
	display_labels.append("False Negative Rate (FNR):")
	# False discovery rate
	FDR = FP/(TP+FP)
	display_data.append(FDR)
	display_labels.append("False Dicovery Rate (FDR): ")

	# Overall accuracy
	ACC = (TP+TN)/(TP+FP+FN+TN) 
	display_data.append(ACC)
	display_labels.append("Overall Accuracy (ACC):")
	
	# Print Final Results
	print '{0:<40} {1:>8}'.format("", ''.join(val+"\t\t" for val in class_names))
	for i, row in enumerate(display_data):
		print '{0:<40} {1:>8}'.format(display_labels[i], ''.join(str(round(val, 2))+"\t\t" for val in row))



