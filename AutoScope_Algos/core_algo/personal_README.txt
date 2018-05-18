*** Valuable information, but not kept up to date and for personal use ***

Execution Instructions: 
+ Pre-crop images vs. real-time cropping: 
++ For pre-cropped images, we create an array of the paths to all the pre-cropped images and annotations. Then, we shuffle the array, and cycle through this array for training/validation. 
++ For real-time cropping of images, we create an array of all the original images. Then, for each batch of training/validation, we pass x images to the cropping generator (where x is the number of main folders). Then, the cropping generator cycles through these x images, cropping a single crop for each image, until the batch size of crops has been generated. 


Possible To Dos: 
+ Particle Class identification: Write a function that determines my accuracy in identifying a particle and identifying the class the particle is in. Currently, I have a function that determines the accuracy of identifying that a particle in the foreground. 
+ Creating generators when needed: Currently, generators are created with the data class. However, another approach is to create the generators with the train and validation functions. Through this, we have more flexibility for the type of generators that are created. 
+ Predict the particle coordinates: Currently, I perform semantic segmentation. However, when seperating foreground/background, a better approach might be to predict particle coordinates. So, I can train my model to do this directly.  
+ Reconfigure class assignments within config: Currently, I configure the class assignments when I create the training data. However, at times, it's convenient to re-configure those classes during training. To do this, I need to create a data structure that maps the old classes to the new classes. 
+ Fixing poor accuracy with semantic segmentation (per-pixel classification): When using semantic segmentation with the real-time cropping approach, we get good segmentation results and poor classification results. This is probably to over-fitting on the training data, where we get great segmentation and classification results. Possible solutions to solve this are: 1) trying pre-cropping and 2) trying rotations (other data augmentation approaches) with real-time cropping. 

Implementation Notes: 
+ The RGB values from the imagenet dataset used for VGG16 is: Blue => 103.939, Green => 116.779, Red => 123.68
+ Per-Image Normalization: Per-channel vs entire image normalization
++ If I normalize across the entire image, I preserve mean differences in color across the channel, which might provide more information. 
++ If I nomralize per channel, the image becomes a greyscale-like image, with color differences very muted. 
+ Use samplewise_center or featurewise_center at your own risk!: investigate first 
++Instead, implement this directly in the preprocessing fuction.
++ First, it's not sure if they work as expected. Second, we might need to use the .fit function to calculate appropriate parameters. 
+ Normalization Notes: 
++ If per-image mean normalization is used, there is no need to use batch or dataset mean normalization. 
++ If dataset normalization is used, there is **much** less of a reason to use batch normalization. 
+ Grayscale Conversion: 
++ For the classification model, we have the option to convert from RGB to grayscale usuing flow_from_directory. This is done through PIL's convert function. 
+ Keras uses PIL: 
++ If you use flow_from_directory, Keras loads your images from a directory with PIL. Thus, the format of each loades image will be (width, height). Be careful. 
