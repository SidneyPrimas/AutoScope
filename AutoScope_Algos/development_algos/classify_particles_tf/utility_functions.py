import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pprint

def get_confusion_matrix(all_truth, all_pred): 
	'Caculate a confusion matrix given the predicted labels and the ground truth labels. '

	classes = all_truth.shape[1] # Get total classes
	truth_class = np.argmax(all_truth, axis=1)
	pred_class = np.argmax(all_pred, axis=1) 
	confusion = np.zeros((classes, classes), dtype=float)
	for num, truth_cl in enumerate(truth_class): 
		confusion[truth_cl, pred_class[num]] += 1
	return confusion


def visualize_filters(filter_struct):
	plt.figure(figsize = (8, 8))
	gs1 = gridspec.GridSpec(8, 8, wspace=0.0, hspace=0.0)

	for i in range(0, np.shape(filter_struct)[3]):
		ax1 = plt.subplot(gs1[i])
		plt.imshow(filter_struct[:, :,0,i])
		plt.axis('off')

	plt.savefig("./classify_particles_tf/data/filter.png",bbox_inches='tight')
	plt.show()


def visualize_image_data(data, labels, step):
	for i in range(0, np.shape(data)[0], step):
		plt.figure(i)
		plt.imshow(data[i, :, :])
		# Labeling Graph
		c = np.argmax(labels[i])
		plt.title("Class %d"%(c))
		plt.xlabel(np.transpose(labels[i]))
	plt.show()


def get_loss_weights(data, params):
	total_images =  sum(data.files_per_class.values())
	weight = np.zeros((params.class_size))
	prob_of_class = np.zeros((params.class_size))
	for key, value in data.files_per_class.iteritems():
		prob_of_class[key] = value/float(total_images)
		weight[key] = params.class_size/(prob_of_class[key] * params.class_size)

	print >> params.log,("Probabability of class based on image distribution: ")
	print >> params.log,(prob_of_class)
	print >> params.log,("Calculated Weights: ")
	print >> params.log,(weight)
	print >> params.log,("\n\n")

	return weight


def print_log_header(particle_data, params):
	""" Print log header, summarizing this model. """
	print >> params.log,("###### HEADER START ###### \n")
	print >> params.log,("Log file: %s")%(params.log.name)
	print >> params.log,("Graph: 2 Convolutional Layers with 2 Convolutions at 3x3")
	print >> params.log,("Equal Images Per Class: (%r)")%(params.equal_images_per_class)
	print >> params.log,("Filter Model Load Path: %s")%(params.filter_load_path)
	print >> params.log,("FC Layer Model Load Path: %s")%(params.fc_layers_load_path)
	print >> params.log,("Number of Classes: %d")%(params.class_size)
	print >> params.log,("Image Dimension: %dx%d")%(params.target_dim, params.target_dim)
	print >> params.log,("Training Set Size: %d")%(len(particle_data.trainlist))
	print >> params.log,("Validation Set Size: %d")%(len(particle_data.validlist))
	print >> params.log, "*** Directory Map ***"
	pprint.pprint(params.directory_map, params.log)
	print >> params.log, "*** Images Per Class ***"
	pprint.pprint(dict(particle_data.files_per_class), params.log)
	print >> params.log,("\n###### HEADER END ###### \n\n\n")

