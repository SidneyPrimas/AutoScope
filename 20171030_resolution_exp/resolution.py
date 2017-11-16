"""
    File name: resolution.py
    Author: Sidney Primas
    Date created: 10/30/2017
    Python Version: 2.7
    Description: Function to calculate the resolution with the 1951 target. 
    Note: Need to think about 1) how to make target straight and 2) using 5MPx vs 8Mpx and 3) how the RGB filter effects the system. 
"""


# Base packages 
import numpy as np
import matplotlib.pyplot as plt
import math

# Import OpenCV
import cv2
# Argparse: For command line python arguments
import argparse

pixel_size = 1.42 #in um (the reason this is 1.42um is that we took the picture at 8MPx, but saved the pictured as 5MPx)

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
	print im_original.shape


	### Plot image on which we can perform cursoe selection of particles
	# Subplot returns both a figure and axes. 
	zoomFig, zoomAx = plt.subplots()
	plt.title("Resolution Test: 1951 Target")
	# Attach the image produced by imshow to zoomAx (axis). imgplot is an AxisImage object. 
	imgplot_axis = zoomAx.imshow(im_original, cmap='gray', interpolation='nearest')
	plt.axis('off')

	### Create ZoomCrop object. Object used to control plot of im_original
	zoom_crop = ZoomCrop(im_original, zoomAx, image_path)
	# On the ZoomCrop object, connect listeners for computer input. 
	zoom_crop.connect(zoomFig)

	# Code runs within ZoomCrop based on interrupts. However, this main program doesn't complete until plot closed. 
	plt.show()

	# Once the plot has been closed, disconnect the keypress listeners. 
	zoom_crop.disconnect(zoomFig)



# Description: Class to interactively manage the image in active_axis. 
class ZoomCrop:
	def __init__(self, img, active_axis, image_path):
		self.img = img
		self.click_pos = (0,0)
		self.release_pos = (0,0)
		self.active_axis = active_axis # Need axis specifically since a figure can have multiple axis. And, we only want to zoom in on a single axis. 
		self.record = False

		### Creat Log (Append data, and flush to file)
		fileName_start = image_path.rfind("/")
		input_file_name = image_path[fileName_start+1:-4]
		input_dir_path = image_path[:fileName_start+1]
		log_file_name = (input_dir_path + input_file_name + "_resolution.txt")
		# Create read/write file that is line buffered (indicated by 1)
		self.log = open(log_file_name,  'w+', 1)

	# Connect to all the events we need
	def connect(self, fig):
		self.cid_press = fig.canvas.mpl_connect('button_release_event', self.onrelease)
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

		print('Click Event: x=%d, y=%d, xdata=%f, ydata=%f' % (event.x, event.y, event.xdata, event.ydata))
		self.click_pos = (event.xdata, event.ydata)

		# Define action during mouse click
	def onrelease(self, event):
		# Don't record point if recording not enabled
		if ~self.record:
			return

		# Indicate if cursor didn't select point on image
		if event.inaxes is None:
			print "Please click on image. Try again"
			return

		print('Release Event: x=%d, y=%d, xdata=%f, ydata=%f' % (event.x, event.y, event.xdata, event.ydata))
		self.release_pos = (event.xdata, event.ydata)

	# Define action during keyboard press
	def onkeypress(self, event): 


		# Select to zoom to click_pos and release_pos coordinates. 
		if event.key == 'z':
			print "Image zoomed"

			# Update zoom in image
			self.active_axis.set_xlim(self.click_pos[0], self.release_pos[0])
			# Flip y-axis (not sure why this is needed) 
			self.active_axis.set_ylim(self.release_pos[1], self.click_pos[1])
			# Update the image (canvas is the currently active canvas)
			event.canvas.draw()


		if event.key == 'd':
			print "Evaluate Distance from Center"
			center = self.img.shape
			distance = math.sqrt((self.click_pos[1] - center[0]/2)**2 + (self.click_pos[0] - center[1]/2)**2)
			print "Shape: (%d, %d)"%(self.img.shape[0],self.img.shape[1])
			print "x-distance: %f"%(self.click_pos[1] - center[0]/2)
			print "y-distance: %f"%(self.click_pos[0] - center[1]/2)
			print "Distance: %f"%(distance)

		# Select to record background
		if event.key == 'e':
			print "Evaluate Chosen Area"
		
			eval_output =  self.img[int(self.click_pos[1]):int(self.release_pos[1]), int(self.click_pos[0]):int(self.release_pos[0])]

			fig = plt.figure()
			fig.patch.set_facecolor('white')
			vertical = eval_output.mean(0)
			contrast = (np.amax(vertical)-np.amin(vertical))/(np.amax(vertical)+np.amin(vertical))
			print >> self.log,("Vertical Contrast: %f")%(contrast)
			print ("Vertical Contrast: %f")%(contrast)
			x_axis = np.arange(float(len(vertical)))*pixel_size
			plt.plot(x_axis,vertical)
			plt.title("Vertical Bars")
			plt.xlabel("Position (um)")
			plt.ylabel("Intensity")


			fig = plt.figure()
			fig.patch.set_facecolor('white')
			horizontal = eval_output.mean(1)
			contrast = (np.amax(horizontal)-np.amin(horizontal))/(np.amax(horizontal)+np.amin(horizontal))
			print >> self.log,("Horizontal Contrast: %f")%(contrast)
			print ("Horizontal Contrast: %f")%(contrast)
			x_axis = np.arange(float(len(horizontal)))*pixel_size
			plt.plot(x_axis,horizontal)
			plt.title("Horizontal Bars")
			plt.xlabel("Position (um)")
			plt.ylabel("Intensity")
			plt.show()
		
		# Toggle if you are recording clicks. 
		if event.key == 'r':
			self.record = ~self.record
			print "Recording Now Set To: %s" % (self.record)



	# Disconnect all the stored connection ids
	def disconnect(self, fig):
		fig.canvas.mpl_disconnect(self.cid_press)
		fig.canvas.mpl_disconnect(self.cid_key)

		# Close figures and log
		self.log.close()

# Command Line Sugar: Calls the main function when file executed from command-line
if __name__ == "__main__":
    main()