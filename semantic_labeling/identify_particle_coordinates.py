# Base packages 
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json 
import os
import cv2
from glob import glob


# Configuration Variables
TARGET_FILE = "./data/20171027/10um/selected1/.bmp"
STEP_SIZE = 700

# image_paths = glob(TARGET_DIR + "*")	
# for target_file in image_paths: 
# if os.path.isdir(target_file): continue

def main(): 

	im_original = cv2.imread(TARGET_FILE, cv2.IMREAD_COLOR)
	im_original = cv2.cvtColor(im_original, cv2.COLOR_BGR2RGB)

	### Plot image on which we can perform cursor selection of particles
	zoomFig, zoomAx = plt.subplots() # Subplot returns both a figure and axes. 
	# Attach the image created by imshow to zoomAx (axis). imgplot is an AxisImage object. 
	imgplot_axis = zoomAx.imshow(im_original, interpolation='nearest')
	plt.title("Reversed Lens Image")
	plt.axis('off')

	### Create ParticleLocation object. Object used to control plot of im_original
	particle_locations = ParticleLocation(STEP_SIZE, im_original, zoomAx, TARGET_FILE)
	# On the ParticleLocation object, connect listeners for computer input. 
	particle_locations.connect(zoomFig)

	### Initialize image to first zoom tile (starting at 0,0)
	zoomAx.set_xlim(0, STEP_SIZE)
	# Flip y-axis so image presented in same format MacOS preview program. 
	zoomAx.set_ylim(STEP_SIZE, 0)
	# Interact with image through ParticleLocation based on interrupts. 
	# The main() program doesn't complete until plot closed. 
	plt.show()

	# Once the plot has been closed, disconnect the keypress listeners. 
	particle_locations.disconnect(zoomFig)


# Description: Class to interactively manage the image in active_axis. 
class ParticleLocation:
	def __init__(self, step_size, img, active_axis, image_path):
		self.x_ref = 0
		self.y_ref = 0
		self.step_size = step_size
		self.img = img
		self.active_axis = active_axis # Need axis specifically since a figure can have multiple axis. And, we only want to zoom in on a single axis. 
		self.particle_list = []
		self.record = False

		### Creat Log (Append data, and flush to file)
		fileName_start = image_path.rfind("/")
		coordinates_file_name = image_path[fileName_start+1:-4]
		coordinates_dir_path = image_path[:fileName_start+1] + "coordinates/"
		if not os.path.exists(coordinates_dir_path):
			os.makedirs(coordinates_dir_path)
		coordinates_file_path = (coordinates_dir_path + coordinates_file_name + "_coordinates.json")
		self.log = open(coordinates_file_path,  'w+', 1) # Create read/write file that is line buffered (indicated by 1)

	# Connect to all the events we need
	def connect(self, fig):
		self.cid_press = fig.canvas.mpl_connect('button_press_event', self.onclick)
		self.cid_key = fig.canvas.mpl_connect('key_press_event', self.onkeypress)


	# Define action during mouse click
	def onclick(self, event):
		# Don't record point if recording not enabled
		if ~self.record:
			return

		# Indicate if cursor didn't select point on image
		if event.inaxes is None:
			print "Please click on image. Try again"
			return

		print('x=%d, y=%d, xdata=%f, ydata=%f' % (event.x, event.y, event.xdata, event.ydata))
		self.particle_list.append([event.xdata, event.ydata])

	# Define action during keyboard press
	def onkeypress(self, event): 

		# 'o': Save the coordinates saved in particle list. 
		if event.key == 'o':
			# JSON encoding used since it's a more versatile solution. 
			json.dump(self.particle_list, self.log)


		# Toggle if you record the clicks
		if event.key == 'r':
			self.record = ~self.record
			print "Recording Now Set To: %s" % (self.record)

		# Resets the zoom region back to (0,0)
		if event.key == 'b':
			# Update attributes
			self.x_ref = 0
			self.y_ref = 0


			# Update zoom in image
			self.active_axis.set_xlim(self.x_ref, self.x_ref + self.step_size)
			# Flip y-axis (not sure why this is needed) 
			self.active_axis.set_ylim(self.y_ref + self.step_size, self.y_ref)
			# Update the image (canvas is the currently active canvas)
			event.canvas.draw()
			

		# 'Enter': Move to next zoomed section of image. 
		if event.key == 'enter': 

			# Update to the next zoom configuration
			next_x = self.x_ref + self.step_size
			next_y = self.y_ref
			
			# End of the image in x direction. Move to next line. 
			if next_x > self.img.shape[1]: 
				next_x = 0 
				next_y = self.y_ref + self.step_size


			# End of the image in y direction. Exit out of the image
			if next_y > self.img.shape[0]:
				print "End of Image"
				
				plt.close()
				return

			
			# Update attributes
			self.x_ref = next_x
			self.y_ref = next_y


			# Update zoom in image
			self.active_axis.set_xlim(self.x_ref, self.x_ref + self.step_size)
			# Flip y-axis (not sure why this is needed) 
			self.active_axis.set_ylim(self.y_ref + self.step_size, self.y_ref)
			# Update the image (canvas is the currently active canvas)
			event.canvas.draw()


	# Disconnect all the stored connection ids
	def disconnect(self, fig):
		fig.canvas.mpl_disconnect(self.cid_press)
		fig.canvas.mpl_disconnect(self.cid_key)

		# Print out coordinates
		print self.particle_list
		json.dump(self.particle_list, self.log)

		# Close figures and log
		self.log.close()



if __name__ == "__main__":
    main()