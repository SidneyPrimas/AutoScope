"""
    File name: visualize_seg.py
    Author: Sidney Primas
    Python Version: 2.7
    Description: Script to help visualize the process of classifying particles. 
"""

import json
import cv2
import glob
from PIL import ImageFont, ImageDraw, Image
import numpy as np

"""
Implementation Notes: 
+ Pil rotation is not efficient: Since the PIL rotation is a pixel-wise transformation, this approach is not efficient. 
++ Instead, implement array based option direclty in numpy. 
"""


reference_img_path = "./data/rbc_base/rbc_img1_display_classes_all.jpg"
input_directory_path = "./data/training_snapshots/rbc_input/"
output_directory_path = "./data/training_snapshots/rbc_output/"


def main():

	image_path_list = glob.glob(input_directory_path + "*")
	image_path_list.sort(key=lambda image_path: int(image_path[image_path.rfind('_')+1:-4]))

	orignal_image = Image.open(reference_img_path).rotate(90, expand=True)
	original_image = np.rot90(orignal_image)
	im_width, im_height = orignal_image.size

	for i, image_path in enumerate(image_path_list):

		image_count = int(image_path[image_path.rfind('_')+1:-4])
		batch_count = image_count/8
		
		current_image = Image.open(image_path)
		current_image = current_image.rotate(90, expand=True)


		# Append the two images. 
		output_im = Image.new('RGB', (im_width*2, im_height))
		output_im.paste(orignal_image, box=None)
		output_im.paste(current_image, (im_width,0))


		# Draw Shapes on image
		draw = ImageDraw.Draw(output_im)
		draw.rectangle([(0,0),(im_width*2,280)], fill=(170,170,170), outline=None)
		draw.rectangle([(im_width,im_height-300),(im_width*2,im_height)], fill=(170,170,170), outline=None)
		draw.line([(im_width,0),(im_width,im_height)], fill=(255,255,255), width=40)


		# Draw text on image
		font_bold = ImageFont.truetype('/Library/Fonts/Arial Bold.ttf', 200)
		font_small = ImageFont.truetype('/Library/Fonts/Arial.ttf', 150)

		draw.text((470, 30), "Gold Standard", font=font_bold, fill=(255,255,255))
		draw.text((im_width+805, 30), "Training", font=font_bold, fill=(255,255,255))

		# Input training cycle info
		draw.text((im_width+200, im_height-230), "Number of Training Cycles:", font=font_small, fill=(255,255,255))
		draw.text((im_width+2050, im_height-230), str(batch_count), font=font_small, fill=(255,255,255))

		# Output Image
		output_path = output_directory_path + "rbc_" + str(image_count) + ".jpg"
		# Resize and write to disk
		output_im.resize((im_width, im_height/2), resample=Image.LANCZOS).save(output_path)


if __name__ == "__main__":
	main()

