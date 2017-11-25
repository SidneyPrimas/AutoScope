### Execution Insutructions ###
+ Initial Folder Structure: The images are stored in folders named after the class. Make sure each class folder only has images, and no subfolders. Update the source data folder, and the split will be done automatically. 
+ Description: The program taks the first 10% of pictures and places them in a "Validation" folder, maintaining the class structure of the original dataset (these are images 0 to X). The rest of the images are places in a "Training" folder 


+ split_dataset.py: Used for the downsampling experiment
+ split_dataset_seg.py: Used for the microscope and reversed lens experiments. 