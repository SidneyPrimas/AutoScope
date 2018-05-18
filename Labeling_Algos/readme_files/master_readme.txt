*** Readme Note ***
Needs to be updated to reflect current state of directory. With that said, parts of the below descriptions are still accurate (and can be used as a starting point). 

###### File Descriptions ######
segment_micro.py: Defines segmentMicroImage funcion that performs segmentation on a single microscope image. Depending on the type of particle, we need to change the variables. Currently, micro_categorization.py is used for the microscope images, and not the reversed lens images. 
micro_categorization.py: Takes a folder full of images, and calls segmentMicroImage on each. Stores the segmented images in the appropriate folder. 
segment_reversed.py: Defines segmentReversedImage funcion that performs segmentation on a single microscope image. Both 6um and 10um particles have the same variables. Currently, reverse_categorization.py is used for the reversed lens images. 
reverse_categorization.py: Takes a folder full of images, and calls segmentReverseImage function on each. Stores the segmented images in the appropriate folder. 
devignette.py: Averages across multiple reversed lens images in order to create a master illumination mask. Then, applies the master illumination mask on a reversed lens image to demonstrate effectiveness. 
segmentation_micro_sandbox: Sandbox used to experiment with algo that segments images from the microscope. The final algo is transferred to segment.py. 
segment_reverse_sandbox: Sandbox used to experiment with algo that segments images from the reversed lens (microbeads).
identify_baf3_coordinates: Identify BAF3 cells with cursor. Outputs a log with all the coordinates stored in JSON format. 