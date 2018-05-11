# Import basic libraries
import numpy as np
import cv2
from glob import glob
import shutil
import sys
import os
import random 
import matplotlib.pyplot as plt
from scipy import stats
import math
import itertools

# import from local libraries
sys.path.insert(0, './urine_particles')
import CNN_functions

""" 
Description: Processes many digiital urine samples through classification algo to determine sensitivity/specificity of classification algo. 

Implementatinos Notes: 
+ Creating digital urine: When creating digital urine: 1) determine the expected particles per HPF (with decimals), 2) multiply by the total of HPFs and 3) round. This allows to give you a better estimate of the exact concentration of the urine. If we round prior to multiplying by HPFs, then you will only get a few discrete solution concentrations. 
+ Running out of particles: When creating a digital solution, if we run out of particles, we reuse particles again. The reason for this: 1) if we run out of particles, we will be so far above the threshold that it won't matter for the results of that specific particle (although it could matter for other particles), 2) different runs of the same repeat particles through the network will give different results, which will reduce variability (this is the whole point of using more than 1 HPF), 3) this is a rare occurence due to the particle distributions. 
+ Drawbacks of this method: 1) We do not include error from segmentation 2) we re-use particles across different digital samples, 3) etc 
+ Distribution Estimation: Since we don't let counts go to inf, we are not a perfect distribtion. However, our range is close enough (way over 95 percent of counts)

Execution Notes: 
+ Before running the script, make sure you selected the correct model in CNN_Functions (since model initialize there)

"""

""" Configuration """
# Main Configuration Parameters
total_solutions_to_run = 209 # indicate the number of solutions to test.
# Reference thresholds are ground truth thresholds. Device thresholds are the custom thresholds set for our system to max spec/sens.
# Anything at given threshold or below is Negative. Anything above is positive. 
thresholds_ref = {'10um': 5, 'other': 0, 'rbc': 3, 'wbc': 5} # A threshold of 0 indicates to exclude the particle.
#thresholds_device = {'10um': 5, 'other': 0, 'rbc': 5, 'wbc': 4}
thresholds_device = {'10um': 5, 'other': 0, 'rbc': 4, 'wbc': 4}

# Files/Folders
input_dir_root = './urine_particles/data/clinical_experiment/image_data/20180225_digital_urine/digital_urine_0p2val/'  
validation_root_dir = input_dir_root + 'validation/'
experimental_root_dir = input_dir_root + 'experiment_folder/'


class_mapping =  {0:'10um', 1:'other', 2:'rbc', 3:'wbc'}
indicator_dict = {'rbc': 'red', 'wbc': 'black', '10um': 'blue', 'other': 'orange'} 
total_HPF_per_solution = 3.0 # Number of HPFs taken to product a results. Global var (not pass through functions.)
min_other_particles_per_HPF = 1 # The minimum number of other particles in each HPF. 

# Distributions are calculated in utility/particle_distributions.py
urine_distributions = {
	'10um': [-0.05702037, -0.03863947, -0.94297963, -0.2990867 ],
	'other':[-0.05702037, -0.03863947, -0.94297963, -0.2990867 ],
	'rbc': [-0.83476613, -0.95886132, -0.16523387, -0.04514418],
	'wbc': [-0.14899523, -0.00707958, -0.85100477, -0.20460091]
}

# Global vars that are updated
reuse_particles_count = 0 #Global var updated during execution 
sample_count = 0



def main():

	# Create CNN model and data model.
	# WARNING: PICK THE CORRECT MODEL IN CNN_FUNCTIONS
	model, data = CNN_functions.initialize_classification_model(log_dir=input_dir_root)
	# Make sure batch_size in configuration correctly setup. 
	if (data.config.batch_size != 1):
		raise ValueError("In digital_urine_experiment, the batch_size needs to be 1. Please update configuration.")

	# Run entire digital urine experiment. 
	reference_all_dict, results_all_dict = run_all_digital_urine_samples(model, data, class_mapping)

	# Create Figures
	summarize_entire_urine_experiment(reference_all_dict, results_all_dict, data.config)



def run_all_digital_urine_samples(model, data, class_mapping):
	""" """
	
	reference_all_dict = {}
	results_all_dict = {}
	for i in range(total_solutions_to_run):
		reference_dict, results_dict = run_single_digital_urine_sample(model, data, class_mapping)

		# Track overall results
		for class_name in thresholds_ref: 

			# Don't include any particles that have a 0 threshold in results. 
			if thresholds_ref[class_name] == 0:
				continue

			# Initialize list in dictionary if necessary
			if class_name not in reference_all_dict:
				reference_all_dict[class_name] = []
				results_all_dict[class_name] = []


			reference_all_dict[class_name].append(reference_dict[class_name])
			results_all_dict[class_name].append(results_dict[class_name])

	return reference_all_dict, results_all_dict


def run_single_digital_urine_sample(model, data, class_mapping):

	global sample_count
	sample_count += 1 
	data.config.logger.info("New Sample: %d", sample_count)

	sample_reference_dict = create_digital_urine_reference(validation_root_dir, class_mapping, urine_distributions)

	create_digital_urine(validation_root_dir, experimental_root_dir, sample_reference_dict)

	results_labels = process_digital_urine(model, data, experimental_root_dir)

	sample_results_dict = calculate_digital_urine_results(sample_reference_dict, results_labels, class_mapping, data.config)

	return sample_reference_dict, sample_results_dict


def create_digital_urine_reference(source_folder, class_mapping, urine_distributions):
	""" """

	urine_reference_dict = {}
	for label, class_name in class_mapping.iteritems():
		count_of_particles_in_class = len(glob(source_folder + class_name + '/*.bmp'))

		# Extract the model paramters
		a1, b1, a2, b2 = urine_distributions[class_name]

		# Obtain the PDF
		counts = np.array(np.linspace(0,40,200))
		pdf = b1*a1*np.exp(b1*counts) + b2*a2*np.exp(b2*counts)

		# Sanity check that the sum of the PDF is within the range
		pdf_sum =  np.trapz(pdf, counts)
		if (pdf_sum > 1.1) or (pdf_sum < 0.85):
			print "Sum of Derivative: %f"%(pdf_sum)
			raise ValueError("The PDF doesn't sum to a number that is sufficiently close to 1.")

		# Generate the urine counts from the distribution. 
		# Eventhough the PDF integrates to 1. We need deriv_y to sum to 1 (not same as integral).
		selected_count = np.random.choice(counts, replace=True, p=pdf/sum(pdf))

		scaled_count = int(round(selected_count*total_HPF_per_solution))
		urine_reference_dict[class_name] = scaled_count 


	urine_reference_dict['other']= urine_reference_dict['other'] + int(min_other_particles_per_HPF*total_HPF_per_solution)
	return urine_reference_dict



def create_digital_urine(source_folder, target_folder, sample_reference_dict):
	"""Creates a sample of digital urine by linking select files from source_folder into target_folder."""

	# Creates target folder (if folder doesn't already exist)
	if (os.path.isdir(target_folder)): 
		shutil.rmtree(target_folder)
	os.makedirs(target_folder + 'images/')

	# Creates digital urine. 
	for urine_class, num_particles in sample_reference_dict.iteritems():

		particle_files = glob(source_folder + urine_class + '/*.bmp')
		random.shuffle(particle_files) # To ensure digital urine is not made through a pre-determined pattern. 
		count_of_particles_in_class = len(particle_files)
		particle_files_iterator = itertools.cycle(particle_files)

		# Flag if the scaled_count is above count of particles in the reference class. 
		if num_particles > count_of_particles_in_class:
			global reuse_particles_count
			reuse_particles_count += 1
			print "Selected Particles: %0.2f"%num_particles
			print "Particles Available: %0.2f"%count_of_particles_in_class

		for index in range(num_particles): 
			particle_file = next(particle_files_iterator)
			split_index =  particle_file.rfind('/') + 1
			target_filename = str(index) + particle_file[split_index:]
			os.link(particle_file, target_folder + 'images/' + target_filename)

		
def process_digital_urine(model, data, target_folder, ):
	"""  """

	# Create necessary data generator
	pred_generator = data.create_custom_prediction_generator(pred_dir_path=target_folder)


	# Predict: Sort images in classes target_folder into classes
	total_cropped_images = len(glob(target_folder + "images/*.bmp"))
	all_pred, all_path_list = data.predict_particle_images(
		model=model, 
		pred_generator=pred_generator, 
		total_batches=total_cropped_images) 


	# Get counts for each of the particle labels
	label_list = np.argmax(all_pred, axis=1)

	return label_list


def calculate_digital_urine_results(reference_dict, results_labels, class_mapping, config):
	""" """
	label_count_dic = CNN_functions.count_labels(results_labels, class_mapping)

	# Translate from numerical labels from CNN to human-readable labels in class_mapping
	results_dict = {}
	for label, class_name in class_mapping.iteritems():
		# Divide counts by the total number of HPFs used. 
		results_dict[class_name] = label_count_dic[label]/total_HPF_per_solution
		reference_dict[class_name] = reference_dict[class_name]/total_HPF_per_solution

	config.logger.info("\t\tReference\tPredicted")
	for class_name in reference_dict:
		config.logger.info("%s:\t\t%0.2f\t\t%0.2f", class_name, reference_dict[class_name], results_dict[class_name])


	return results_dict

def summarize_entire_urine_experiment(reference_all_dict, results_all_dict, config):
	""" """

	# Create scatter of results
	fig = plt.figure()
	reference_all_list = []
	results_all_list = []
	for class_name in reference_all_dict: 
		reference_all_list.extend(reference_all_dict[class_name])
		results_all_list.extend(results_all_dict[class_name])

		# Update Scatter Plot
		plt.scatter(reference_all_dict[class_name], results_all_dict[class_name], label=class_name, color=indicator_dict[class_name], s=60, facecolors='none')

		# Print Per Particle Best Fit
		slope, intercept, r_value, p_value, stderr = stats.linregress(reference_all_dict[class_name], results_all_dict[class_name])
		r_squared = r_value**2
		config.logger.info("** %s **: slope: %f, intercept: %f, r_squared: %f, p_value: %f, stderr: %f",class_name, slope, intercept, r_squared, p_value, stderr)
	

	# Add the best fit line
	slope, intercept, r_value, p_value, stderr = stats.linregress(reference_all_list, results_all_list)
	r_squared = r_value**2
	config.logger.info("** All **: slope: %f, intercept: %f, r_squared: %f, p_value: %f, stderr: %f",slope, intercept, r_squared, p_value, stderr)
	fitted_line_range = int(max(reference_all_list))+1
	fitted_line = [(intercept + slope*value) for value in range(fitted_line_range)]
	fitted_legend_str = ("Fitted (r^2: %0.03f)")%(r_squared)
	plt.plot(range(fitted_line_range), fitted_line, color='gray', label=fitted_legend_str)

	# Plot configurations 
	fig.patch.set_facecolor('white')
	plt.legend(loc='upper left', prop={'size':12}, frameon=True)
	plt.xlabel("Reference Counts (Per HPF)", fontsize="15")
	plt.ylabel("AutoScope Counts (Per HPF)", fontsize="15")


	# Print number of times particles had to be reused while making a digital solution
	config.logger.info("Count of when particles had to be reused: %d", reuse_particles_count)

	# Create hisograms
	for class_name in reference_all_dict: 

		# Calculate PDF
		# Extract the model paramters
		a1, b1, a2, b2 = urine_distributions[class_name]

		# Obtain the PDF
		counts = np.array(np.linspace(0,40,200))
		pdf = b1*a1*np.exp(b1*counts) + b2*a2*np.exp(b2*counts)

		#Create graphs
		fig = plt.figure()
		n, bins, patches = plt.hist(reference_all_dict[class_name], bins=40, range=(0,40), normed=True, facecolor='green', alpha=0.75)
		plt.plot(counts, pdf, color=indicator_dict[class_name])
		# Plot configuration
		fig.patch.set_facecolor('white')
		plt.title(class_name)
		plt.legend(loc='upper left', prop={'size':12}, frameon=True)
		plt.xlabel("Reference Particle Counts (Per HPF)", fontsize="15")
		plt.ylabel("Probablity Distribution", fontsize="15")
		axes = fig.axes[0]
		axes.set_xlim([0,30])
		axes.set_ylim([0,1])

	# Calculate Sensitivity + Specificity 
	for class_name in reference_all_dict: 

		# Determine if solution is labeled Positive/Negative
		thresh_reference_list = np.array(reference_all_dict[class_name]) > thresholds_ref[class_name]
		thresh_results_list = np.array(results_all_dict[class_name]) > thresholds_device[class_name]

		# Calculate confusion Matrix
		TP = 0.0
		FP = 0.0
		TN = 0.0
		FN = 0.0
		Total = 0.0
		for index, thresh_reference in enumerate(thresh_reference_list):
			thresh_results = thresh_results_list[index]

			Total += 1
			if (thresh_reference == True) and (thresh_results == True):
				TP += 1
			if (thresh_reference == True) and (thresh_results == False):
				FN += 1
			if (thresh_reference == False) and (thresh_results == True):
				FP += 1
			if (thresh_reference == False) and (thresh_results == False):
				TN += 1

		# Calculate and display results. 
		display_data = []
		display_labels = []

		# Sensitivity, hit rate, recall, or true positive rate
		TPR = 'N/A' if ((TP+FN) == 0) else TP/(TP+FN)
		display_data.append(TPR)
		display_labels.append("Sensitivity (TPR):")
		# Specificity or true negative rate
		TNR = 'N/A' if ((TN+FP) == 0) else TN/(TN+FP) 
		display_data.append(TNR)
		display_labels.append("Specificity (TNR):")
		# Precision or positive predictive value
		PPV = 'N/A' if ((TP+FP)==0) else TP/(TP+FP)
		display_data.append(PPV)
		display_labels.append("Precision (PPV):")
		# Negative predictive value
		NPV = 'N/A' if ((TN+FN)==0) else TN/(TN+FN)
		display_data.append(NPV)
		display_labels.append("Negative Predictive Value (NPV):")
		# Fall out or false positive rate
		FPR = 'N/A' if ((FP+TN)==0) else FP/(FP+TN)
		display_data.append(FPR)
		display_labels.append("False Positive Rate (FPR):")
		# False negative rate
		FNR = 'N/A' if ((TP+FN)==0) else FN/(TP+FN)
		display_data.append(FNR)
		display_labels.append("False Negative Rate (FNR):")
		# False discovery rate
		FDR = 'N/A' if ((TP+FP)==0) else FP/(TP+FP)
		display_data.append(FDR)
		display_labels.append("False Dicovery Rate (FDR): ")
		# Overall accuracy
		ACC = 'N/A' if ((TP+FP+FN+TN)  == 0) else (TP+TN)/(TP+FP+FN+TN) 
		display_data.append(ACC)
		display_labels.append("Overall Accuracy (ACC):")

	
		# Print Final Results
		config.logger.info("******** Results for : %s ********", class_name)
		for i, row in enumerate(display_data):
			output_results = '{0:<40} {1:>8}'.format(display_labels[i], ''.join(str(round(display_data[i], 2))))
			config.logger.info(output_results)


	plt.show()



if __name__ == "__main__":
	main()