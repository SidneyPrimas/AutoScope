"""
    File name: identify_baf3_coordinates.py
    Author: Sidney Primas
    Date created: 06/28/2017
    Python Version: 2.7
    Description: Identify baf3 coordinates using mouse cursor. Currently, we are not able to do automated segmentation of BAF3 cells.
"""


# Base packages 
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json 

# Import OpenCV
import cv2
# Argparse: For command line python arguments
import argparse

# Configurable Variables
step_size = 500 
mask_path = "./data/20170425/reference/illumination_mask.jpg"

def main():


	### Construct the argument parse and parse the arguments
	# Obtain the BAF3 Image
	# -i: Path to image
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True, help="path to input image")
	args = vars(ap.parse_args())
	image_path = args["image"]

	### Load Image: Ensure image is in grayscale. 
	im_original = cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE)


	### Preprocess the Image: Including compensation for illumination 
	### Implement Illumination Compensation
	im_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
	# Convert illumination mask into floating point format (to allow for compensation)
	im_mask = im_mask * 1.0
	# Compenstate with illumination mask. Still preserves floating point format. 
	im_compensated = np.divide(im_original, im_mask)
	# cv2.convertScaleAbs(src[, dst[, alpha[, beta]]]) -> dst
	# Converts matrix to an 8-bit data type after scaling by alpha. 
	im_compensated = cv2.convertScaleAbs(im_compensated, alpha = (255.0/np.amax(im_compensated)))


	### Plot image on which we can perform cursoe selection of particles
	# Subplot returns both a figure and axes. 
	zoomFig, zoomAx = plt.subplots()
	plt.title("Reversed Lens BAF3 Image")
	# Attach the image produced by imshow to zoomAx (axis). imgplot is an AxisImage object. 
	imgplot_axis = zoomAx.imshow(im_compensated, cmap='gray', interpolation='nearest')
	plt.axis('off')

	### Create ZoomCrop object. Object used to control plot of im_original
	zoom_crop = ZoomCrop(step_size, im_compensated, zoomAx, image_path)
	# On the ZoomCrop object, connect listeners for computer input. 
	zoom_crop.connect(zoomFig)

	### Initialize image to first zoom tile (starting at 0,0)
	zoomAx.set_xlim(0, step_size)
	# Flip y-axis so image presented in expected format(not sure why this is needed) 
	zoomAx.set_ylim(step_size, 0)
	# Code runs within ZoomCrop based on interrupts. However, this main program doesn't complete until plot closed. 
	plt.show()

	# Once the plot has been closed, disconnect the keypress listeners. 
	zoom_crop.disconnect(zoomFig)


# Description: Class to interactively manage the image in active_axis. 
class ZoomCrop:
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
		input_file_name = image_path[fileName_start+1:-4]
		input_dir_path = image_path[:fileName_start+1]
		log_file_name = (input_dir_path + "coordinates/" + input_file_name + "_coordinates.json")
		# Create read/write file that is line buffered (indicated by 1)
		self.log = open(log_file_name,  'w+', 1)

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
		json.dump(self.particle_list, self.log)

		# Close figures and log
		self.log.close()


# Command Line Sugar: Calls the main function when file executed from command-line
if __name__ == "__main__":
    main()