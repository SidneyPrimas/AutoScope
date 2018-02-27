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

Execution Notes: 
+ Creating digital urine: When creating digital urine: 1) determine the expected particles per HPF (with decimals), 2) multiply by the total of HPFs and 3) round. This allows to give you a better estimate of the exact concentration of the urine. If we round prior to multiplying by HPFs, then you will only get a few discrete solution concentrations. 
+ Running out of particles: When creating a digital solution, if we run out of particles, we reuse particles again. The reason for this: 1) if we run out of particles, we will be so far above the threshold that it won't matter for the results of that specific particle (although it could matter for other particles), 2) different runs of the same repeat particles through the network will give different results, which will reduce variability (this is the whole point of using more than 1 HPF), 3) this is a rare occurence due to the particle distributions. 
+ Drawbacks of this method: 1) We do not include error from segmentation 2) we re-use particles across different digital samples, 3) etc 
"""

""" Configuration """
# Main Configuration Parameters
total_solutions_to_run = 100 # indicate the number of solutions to test.

# Files/Folders
input_dir_root = './urine_particles/data/clinical_experiment/image_data/20180225_digital_urine/digital_urine_v1/'  
validation_root_dir = input_dir_root + 'validation/'
experimental_root_dir = input_dir_root + 'experiment_folder/'


class_mapping =  {0:'10um', 1:'other', 2:'rbc', 3:'wbc'}
# Anything at given threshold or below is Negative. Anything above is positive. 
thresholds = {'10um': 5, 'other': 0, 'rbc': 5, 'wbc': 5} # A threshold of 0 indicates to exclude the particle.
indicator_dict = {'rbc': 'red', 'wbc': 'black', '10um': 'blue', 'other': 'orange'} 
total_HPF_per_solution = 2.0 # Number of HPFs taken to product a results. Global var (not pass through functions.)
min_other_particles_per_HPF = 1 # The minimum number of other particles in each HPF. 

# Distributions are calculated in utility/particle_distributions.py
urine_distributions = {
	'10um': [ 0.2294182,  -0.13857566,  0.7705818,  -1.01251156],
	'other': [ 0.2294182,  -0.13857566,  0.7705818,  -1.01251156],
	'rbc': [ 0.82998251, -0.95769342,  0.1652328,  -0.04514385],
	'wbc': [ 0.6399102,  -0.44225141,  0.3600898,  -0.0301516 ]
}

reuse_particles_count = 0 #Global var updated during execution 



def main():

	# Create CNN model and data model.
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
		for class_name in thresholds: 

			# Don't include any particles that have a 0 threshold in results. 
			if thresholds[class_name] == 0:
				continue

			# Initialize list in dictionary if necessary
			if class_name not in reference_all_dict:
				reference_all_dict[class_name] = []
				results_all_dict[class_name] = []


			reference_all_dict[class_name].append(reference_dict[class_name])
			results_all_dict[class_name].append(results_dict[class_name])

	return reference_all_dict, results_all_dict


def run_single_digital_urine_sample(model, data, class_mapping):


	data.config.logger.info("New Sample: ")

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
		counts = np.array(np.linspace(0,50,200))
		pdf = -b1*a1*np.exp(b1*counts) + -b2*a2*np.exp(b2*counts)

		# Sanity check that the sum of the PDF is within the range
		pdf_sum =  np.trapz(pdf, counts)
		if (pdf_sum > 1.1) or (pdf_sum < 0.9):
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
		plt.scatter(reference_all_dict[class_name], results_all_dict[class_name], label=class_name, color=indicator_dict[class_name], s=60, facecolors='none')
	

	# Add the best fit line
	slope, intercept, r_value, p_value, stderr = stats.linregress(reference_all_list, results_all_list)
	r_squared = r_value**2
	config.logger.info("slope: %f, intercept: %f, r_squared: %f, p_value: %f, stderr: %f",slope, intercept, r_squared, p_value, stderr)
	fitted_line_range = int(max(reference_all_list))+1
	fitted_line = [(intercept + slope*value) for value in range(fitted_line_range)]
	fitted_legend_str = ("Fitted (r^2: %0.03f)")%(r_squared)
	plt.plot(range(fitted_line_range), fitted_line, color='gray', label=fitted_legend_str)

	# Plot configurations 
	fig.patch.set_facecolor('white')
	plt.legend(loc='upper left', prop={'size':12}, frameon=True)
	plt.xlabel("Reference Results (Per HPF)", fontsize="20")
	plt.ylabel("Classification Results (Per HPF)", fontsize="20")


	# Print number of times particles had to be reused while making a digital solution
	config.logger.info("Count of when particles had to be reused: %d", reuse_particles_count)

	# Create hisograms
	for class_name in reference_all_dict: 

		# Calculate PDF
		# Extract the model paramters
		a1, b1, a2, b2 = urine_distributions[class_name]

		# Obtain the PDF
		counts = np.array(np.linspace(0,50,200))
		pdf = -b1*a1*np.exp(b1*counts) + -b2*a2*np.exp(b2*counts)

		#Create graphs
		fig = plt.figure()
		n, bins, patches = plt.hist(reference_all_dict[class_name], bins=20, normed=True, facecolor='green', alpha=0.75)
		plt.plot(counts, pdf)
		# Plot configuration
		fig.patch.set_facecolor('white')
		plt.legend(loc='upper left', prop={'size':12}, frameon=True)
		plt.xlabel("Reference Results (Per HPF)", fontsize="20")
		plt.ylabel("Classification Results (Per HPF)", fontsize="20")
		axes = fig.axes[0]
		axes.set_xlim([0,30])

	# Calculate Sensitivity + Specificity 
	for class_name in reference_all_dict: 

		# Determine if solution is labeled Positive/Negative
		thresh_reference_list = np.array(reference_all_dict[class_name]) > thresholds[class_name]
		thresh_results_list = np.array(results_all_dict[class_name]) > thresholds[class_name]

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

		# Display results
		config.logger.info("Results for : %s", class_name)
		# Sensitivity, hit rate, recall, or true positive rate
		TPR = 'N/A' if ((TP+FN) == 0) else TP/(TP+FN)
		config.logger.info("Sensitivity (TPR): %s", str(TPR))
		# Specificity or true negative rate
		TNR = 'N/A' if ((TN+FP) == 0) else TN/(TN+FP) 
		config.logger.info("Specificity (TNR): %s", str(TNR))


	plt.show()



if __name__ == "__main__":
	main()