import json

particle_list = []
input_file = "./data/20171027/rbc/selected1/raw_input.txt"
with open(input_file) as f:
	for line in f:
		data_array = line.strip().split(',')
		xdata = float(data_array[2].split('=')[-1])
		ydata = float(data_array[3].split('=')[-1])
		particle_list.append([xdata, ydata])


# Dump data to json file
coordinate_file_output = "./data/20171027/rbc/selected1/coordinates/img2_light_coordinates.json"
log = open(coordinate_file_output,  'w+', 1) # Create read/write file that is line buffered (indicated by 1)
json.dump(particle_list, log)
log.close()


