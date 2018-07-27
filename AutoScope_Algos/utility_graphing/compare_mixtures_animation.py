import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import re
import sys
import pprint
from random import shuffle
from matplotlib.patches import Circle

"""
Description: Used to animate graphs produces by compare_mixtures.py
"""

# Import homebrew functions
base_directory = "./core_algo/data/clinical_experiment/log/20180205_training_plus/particleCount_cloud_results/20180205_final_model_highAug_original/"
sys.path.append(base_directory)
import config 



""" Configuration Variables """
# Converts from Primas unit to HPF unit. 
hpf_area = 0.196349541 # in mm^2
primas_area = 10.1568 # in mm^2
k_primas_to_HPF = hpf_area/float(primas_area) 

indicator_dict = {'rbc': 'red', 'wbc': 'black', '10um': 'blue' }

title = ""



def main():


	results_dict = process_log()

	target_metrics_dict = calculate_solution_metrics()

	# Debug
	pp = pprint.PrettyPrinter(indent=1)
	#pp.pprint(results_dict)
	#pp.pprint(target_metrics_dict)

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
			if (log_avg) and (img_name != 'final'): 
				continue
			# Handles log_avg case. Skips final average if looking at results for each image. 
			if (not log_avg) and (img_name == 'final'): 
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


	plot_dict_results_animation(target_metrics, results_metrics)



def plot_dict_results_animation(target_metrics, results_metrics):


	# Restructure data to plot single data point at a time [tuple(class_name, target_val, result_val)]
	animation_data_struct = []
	for class_name, target_list in target_metrics.iteritems():
		for i, target_val in enumerate(target_list):
			animation_data_struct.append((class_name, target_val, results_metrics[class_name][i]))


	# Shuffle to randomly populate data points
	shuffle(animation_data_struct)

	fig = plt.figure(figsize = (10,6), dpi=250)
	fig.patch.set_facecolor('white')
	plt.xlabel("Reference Results (%)", fontsize="20")
	plt.ylabel("AutoScope Results (%)", fontsize="20")
	axes = fig.axes[0]
	axes.set_ylim([-5,105])
	axes.set_xlim([-5,105])

	# Force Legend from start
	plt.plot([],[], 'or', label='Red Blood Cells (RBCs)', markersize=10)
	plt.plot([],[], 'ob', label='10um Microbeads', markersize=10)
	plt.plot([],[], 'ok', label='White Blood Cells (WBCs)', mfc='none', markersize=10)
	leg = axes.legend(loc='upper left', prop={'size':14}, frameon=False)
	# Change color of text within legend
	for line, text in zip(leg.get_lines(), leg.get_texts()):
		text.set_color(line.get_color())

	# Save initial version 
	output_path = './data_output/20180727_results_animation/00.png'
	fig.savefig(output_path)

	for i, data_point in enumerate(animation_data_struct):
		# Add point to scatter plot
		class_name = data_point[0]
		if (class_name == 'wbc'):
			plt.scatter(data_point[1], data_point[2], label=class_name, color=indicator_dict[class_name], s=120, facecolors='none')
		else: 
			plt.scatter(data_point[1], data_point[2], label=class_name, color=indicator_dict[class_name], s=120)

		output_path = './data_output/20180727_results_animation/' + str(i) + '.png'
		fig.savefig(output_path)



	# Add fitted line
	intercept = 3.138772
	slope = 0.905837
	r_squared = 0.980137 
	fitted_legend_str = ("R-Squared: %0.03f")%(r_squared)
	plt.text(65, 0, fitted_legend_str, fontsize=20)
	fitted_line = [(intercept + slope*value) for value in range(100)]
	plt.plot(range(100), fitted_line, color='gray', label=fitted_legend_str)
	output_path = './data_output/20180727_results_animation/final.png'
	fig.savefig(output_path)



if __name__ == "__main__":
	main()