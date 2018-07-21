"""
    File name: visualize_classification.py
    Author: Sidney Primas
    Python Version: 2.7
    Description: Script to help visualize the process of classifying particles. 
"""

import json
import cv2
from PIL import ImageFont, ImageDraw, Image

image_path = "./data/sol7/img1.bmp"
label_path = "./data/sol7/img1_crops/debug_output/sol7_label_info.json"
output_directory_path = "./data/sol7/classify_snapshots/"

class_mapping =  {0:'10um', 1:'other', 2:'rbc', 3:'wbc'}

colors = [(255,0,0),(255,255,255),(0,0,255),(0,0,0)]


def main():
	with open(label_path, 'r') as fp:
		label_data = json.load(fp)

	original_img = cv2.imread(image_path)

	particle_count = {'rbc':0, 'wbc':0, '10um':0, 'other':0}
	output_log = open(output_directory_path + 'counts.log', 'w')

	# Write out original state
	update_and_output_image(original_img, particle_count, count=0)
	output_log.write("%d.jpg: %s\n"%(0, particle_count))


	count = 0
	for particle_info in label_data['img1']:
		print particle_info
		count += 1 # Number of particle locations processed. 

		# Update Image with indicator
		circle_centroid = (int(particle_info['center'][0]), int(particle_info['center'][1]))
		cv2.circle(original_img, 
			center=circle_centroid, 
			radius=32, 
			color=particle_info['color'], 
			thickness=3)

		# Update total counts
		particle_count[class_mapping[particle_info['label']]] += 1 

		# Output Data (write to file)
		if (count%5 == 0):

			# Prepare and write image
			update_and_output_image(original_img, particle_count, count)
			# Write log file
			output_log.write("%d.jpg: %s\n"%(count, particle_count))

	
	# Write output of final state.
	update_and_output_image(original_img, particle_count, count)
	output_log.write("%d.jpg: %s\n"%(count, particle_count))

	# Close the file
	output_log.close()


def update_and_output_image(original_img, particle_count, count):
	"""
	Updates text on original image. Writes output image to file. 
	"""

	# Write Image
	cv2_im_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
	pil_im = Image.fromarray(cv2_im_rgb)
	draw = ImageDraw.Draw(pil_im)

	# Draw the rectangle
	draw.rectangle([(0,0),(1050,725)], fill=(170,170,170), outline=None)
	
	# Choose a font
	font_normal = ImageFont.truetype('/Library/Fonts/Arial.ttf', 100)
	font_bold = ImageFont.truetype('/Library/Fonts/Arial Bold.ttf', 100)

	# Draw the text
	title = "PARTICLE COUNTS"
	draw.text((50, 50), title, font=font_bold, fill=(0, 0, 0))
	text_matrix = 	[
						["RBC",		str(particle_count['rbc']),		"/105"],
						["WBC",		str(particle_count['wbc']), 	"/35"],
						["10um",	str(particle_count['10um']), 	"/19"],
						["Other",	str(particle_count['other']),	]
					]

	# Print text to image. 
	for row in range(len(text_matrix)):
		for col, output_str in enumerate(text_matrix[row]):

			# Initial
			col_loc = col*300+50
			row_loc = row*130+190
			font = font_normal
			# Handle exceptions
			if (col == 1):
				col_loc = 450
			if (col == 0):
				font = font_bold

			# Draw Text
			draw.text((col_loc, row_loc), output_str, font=font, fill=colors[row])


	# Output Image 
	output_path = output_directory_path + str(count) + ".jpg"
	

	# Save (or resize and save)
	#pil_im.save(output_path)
	im_width, im_height = pil_im.size
	pil_im.resize((im_width/2, im_height/2), resample=Image.LANCZOS).save(output_path)



if __name__ == "__main__":
	main()

