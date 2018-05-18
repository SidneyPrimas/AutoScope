*** Valuable information, but out of date ***

 classify_imagenet.py
+ In this implementation, I used cv2 for image importing and image manipulation. In the future, use PIL since the network is trained on PIL configuration and methods. cv2 uses slightly different configurations: BGR instead of RGB, switches width/height and has different resizing techniques. These differences make a large difference in the output. 

train_iris.py Notes: 
+ Greyscale Issue: Imagenet VGG models use 3 channels while the IRIS dataset is only greyscale. The issue is with uploading the pre-trained weights. The weights on the first convolutoinal layer assume 3x3 kernels for each input channel. (The weights are not dependent on the image size but ARE dependent on the image channels.) So, there is a weight mismatch when we are loading the pre-trained weights. The solutions are: 1) replicate the greyscale to RGB, 2) average the three 3x3 kernels to a single 3x3 kernel (manipulate the weights directly) or 3) add an additional layer. Good explanation of approach: http://forums.fast.ai/t/black-and-white-images-on-vgg16/2479/3

train_iris.py To Do:
+ Greyscale Issue: Experiment with better ways to handle the greyscale issue. 
+ Low Level Features: Experiment with removing some of the higher-level convolutional layers to get more direct access to the lower level features (which is all I need).  
+ Normailze Dataset: Create a function that normalizes the dataset (including mean and stdev normalization)
+ High Level Organization: Create a data pre-processing script that will focus on prepping the data for training (including creating directories and npz files). This will be completely seperate from the train file. 
+ Write train from scratch so that we can have more control over how we train our system. 

Configuration Notes: 
+ Instance vs. Class Variable for configuration: At the highest leve, an instance variable belongs to the instance, and is initialized with the constructor. And, a class variable belongs to the class. When changing a class variable through the instance, we change the class variable only for the instance in question. When changing the class variable through the Class (by calling Class.x = 0), we change the class variable for all instances that haven't been changed yet (since instances that have been changed point to a new location in memory). For our purposes, we will create class variables. The reason is that each configuration will only have a single instance, and it's easier to create class variables. For instance variables, we should write functions to update them (which takes time)


Organization Notes: 
+ Functions to train specific layers: Models should be created in model factory. Manipulating which layers are trainable should be done in the training and validation files. 
+ Organization Strategy: First, write all the functions that are needed. Then, organize the functions into classes. 


Questions: 
+ Does flow_from_directory provide batches that have equal images across classes.? Or, do we provide totally random batches? 
++ My guess is that it's random. 