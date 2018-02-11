"""
    File name: compare_mixtures.py
    Author: Sidney Primas
    Python Version: 2.7
    Description: Used to compare classification results across different solution mixtures. 
"""

import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import re
import sys
import pprint

# Import homebrew functions
base_directory = "./urine_particles/data/clinical_experiment/log/20180205_training_plus/particleCount_cloud_results/notEqual_withPos/"
sys.path.append(base_directory)
import config 



""" Configuration Variables """
# Converts from Primas unit to HPF unit. 
hpf_area = 0.196349541 # in mm^2
primas_area = 10.1568 # in mm^2
k_primas_to_HPF = hpf_area/float(primas_area) 

title = ""



def main():


	results_dict = process_log()

	target_metrics_dict = calculate_solution_metrics()

	# Debug
	pp = pprint.PrettyPrinter(indent=1)
	pp.pprint(results_dict)
	pp.pprint(target_metrics_dict)

	plot_results(results_dict, target_metrics_dict, title)



def process_log():
	"""
	Description: Parses a log into the results_dict, which summarizes metrics within the log. 
	Returns 
	results_dict: A dictionary of dictionaries. sol_name (usually a specific solution) => image names => class name => dict of metrics {perPrimas, perHPF, percent}
	+ Each log has a 'final' image_name that summarizes the results from all images.
	"""
	# Variables that aggregate data across all logs. 
	results_dict = {}

	### Extract relevant data from the logs ###
	for index, file_name in enumerate(config.logs['filename']): 

		# Variables that aggregate data from single log (usually a specific solution)
		sol_name = config.logs['sol_name'][index]
		results_dict[sol_name] = {}

		# Parse through log to pull out summary results. 
		log_path = config.log_directory + file_name
		with open(log_path) as f:

			img_name = None # Only start processing once results for first image found. 

			# Process each line of log
			for line in f:

				# Identify results section. Update dictionary to expect new results. 
				if (line.find('Results for:') >= 0):
					img_name = line.split('/')[-3]
					results_dict[sol_name][img_name] = {}
					continue

				# Identify final results section. Update dictionary to expect final results. 
				if (line.find('Final Results Summary:') >=0):
					img_name = 'final'
					results_dict[sol_name][img_name] = {}
					continue
					

				# Skip lines until the first 'Results Section' is identified
				if (img_name == None):
					continue
				
				# Identify and extract metrics for each particle class 
				for class_name in config.class_mapping:
					if (line.find(class_name + '\t') >= 0) and img_name:
						raw_metric_outputs = line.split()
						metric_outputs = [float(data.replace('%','').replace('N/A','0')) for data in raw_metric_outputs[2:]]
						metric_dict = {
							'perPrimas': metric_outputs[0], 
							'perHPF': metric_outputs[1], 
							'percent': metric_outputs[2]}
						results_dict[sol_name][img_name][class_name] = metric_dict
						continue
												
	return results_dict



def calculate_solution_metrics():
	"""
	Description: Calculates the expected results for each log (each log is usually a solution)
	Returns 
	target_metrics_all: Dictionary of expected results. sol_name => class_name => dict of metrics {perPrimas, perHPF, percent}
	"""
	# Variables 
	target_metrics_all = {}

	# Loops over expected results for each log
	for index, sol_name in enumerate(config.logs['sol_name']): 

		# Obtain total expected particles in mixture
		expected_perPrimas = config.logs['expected_results'][index]
		total_particles = sum(expected_perPrimas.values())

		# Update target_metrics_all with correct metrics
		target_metrics_all[sol_name] = {}
		for class_name in expected_perPrimas:
			perPrimas = expected_perPrimas[class_name]
			perHPF = perPrimas*k_primas_to_HPF
			percent = 100*perPrimas/float(total_particles)
			metric_dict = {'perPrimas': perPrimas, 'perHPF': perHPF, 'percent': percent}
			target_metrics_all[sol_name][class_name] = metric_dict

	return target_metrics_all



def plot_results(results_dict, target_metrics_dict, title, log_avg=True, graph_classes=True, metric='percent'):

	# Aggregate results
	results_metrics = {} if graph_classes else []
	target_metrics = {} if graph_classes else []
	for sol_name in results_dict:
		for img_name in results_dict[sol_name]:
			
			# Handles log_avg case. only aggregate the retuls for averages in 'final' 
			if (log_avg): 
				if(img_name != 'final'):
					continue

			for class_name in results_dict[sol_name][img_name]:
				# Other class not relevant for comparing to expected metrics. 
				if class_name == 'other':
					continue

				# Handles graph_classes flag by splitting results into classes. 
				if (graph_classes):
					if class_name not in results_metrics:
						results_metrics[class_name] = []
						target_metrics[class_name] = []

					results_metrics[class_name].append(results_dict[sol_name][img_name][class_name][metric])
					target_metrics[class_name].append(target_metrics_dict[sol_name][class_name][metric])

				# Shows all results
				else: 
					results_metrics.append(results_dict[sol_name][img_name][class_name][metric])
					target_metrics.append(target_metrics_dict[sol_name][class_name][metric])


	if (graph_classes):
		plot_dict_results(target_metrics, results_metrics)
	else: 
		plot_list_results(target_metrics, results_metrics)

def plot_dict_results(target_metrics, results_metrics):
	
	fig = plt.figure()

	# Plot each class with a different color
	for class_name in results_metrics: 

		# Calculate relevant statistics 
		slope, intercept, r_value, p_value, _ = stats.linregress(target_metrics[class_name], results_metrics[class_name])
		r_squared = r_value**2
		print ("slope: %f, intercept: %f, r_squared: %f, p_value: %f")%(slope, intercept, r_squared, p_value)

		# Plot
		fitted_line = [(intercept + slope*value) for value in target_metrics[class_name]]
		fitted_legend_str = ("%s (r^2: %0.03f)")%(class_name, r_squared)
		plt.plot(target_metrics[class_name], fitted_line, label=fitted_legend_str)
		plt.scatter(target_metrics[class_name], results_metrics[class_name], label=class_name)


	# Plot characteristics
	plt.title(title)
	fig.patch.set_facecolor('white')
	plt.legend(loc='upper left', prop={'size':12}, frameon=True)
	plt.xlabel("Expected Proportion (%)", fontsize="20")
	plt.ylabel("Measured Proportion (%)", fontsize="20")
	# axes = fig.axes[0]
	# axes.set_ylim([0,100])
	# axes.set_xlim([0,100])
	plt.show()

def plot_list_results(target_metrics, results_metrics):
	

	fig = plt.figure()

	# Calculate relevant statistics 
	slope, intercept, r_value, p_value, _ = stats.linregress(target_metrics, results_metrics)
	r_squared = r_value**2
	print ("slope: %f, intercept: %f, r_squared: %f, p_value: %f")%(slope, intercept, r_squared, p_value)

	# Plot
	plt.title(title)
	plt.scatter(target_metrics, results_metrics, label="metric")
	fitted_line = [(intercept + slope*value) for value in target_metrics]
	fitted_legend_str = ("Fitted (r^2: %0.03f)")%(r_squared)
	plt.plot(target_metrics, fitted_line, 'r', label=fitted_legend_str)
	fig.patch.set_facecolor('white')
	plt.legend(loc='lower right', prop={'size':12}, frameon=True)
	plt.xlabel("Expected Proportion (%)", fontsize="20")
	plt.ylabel("Measured Proportion (%)", fontsize="20")
	# axes = fig.axes[0]
	# axes.set_ylim([0,100])
	# axes.set_xlim([0,100])
	plt.show()



if __name__ == "__main__":
	main()