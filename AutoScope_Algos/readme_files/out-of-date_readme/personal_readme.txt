*** Valuable information, but not kept up to date and for personal use ***

Execution: 
+ Use particles_ML_model.py to train and validate the neural net. 
+ Use split_dataset.py to split particle images into training and validation folders. Execute from root folder. 
+ Use resample.py to resample the reference images to the correct resolution (analog and digital resolution). Execute from root folder. 


QUESTION
+ What is the best way to train the model: in what distribution should I use the training images. 

TO DO
General: 
+ Add readme's to the models that are currently being trained (including their setup)
+ Change the image size to 64x64 (and adjust the model accordingly)
+ Implement python line options (for the most important options)
+ Implement GPU computing
+ Use TF GUI to monitor progress
+ Built the accuracy grid (showing in detail false-positive and false-negative)
+ Complete image processing tutorial on TensorFlow Website
+ Seperate Model into function. This allows for seperate code for training and validation. 
+ Add script that shows how we converge to the results


particles_ML_model.py
+ Implement a better saving scheme (currently I only save at the end of an iteration)
+ see file for further todos 
+ Important: Need an improved approach for splitting validation and training. Need away to record which are the validation images and which are the training. 1) create a file to store the training images 2) pre-create folders with training and validation. 
+ Ideal Validation setup: Each class has equal number of images. This is a true test of performance (instead of depending on the bias term)
+ Important: If model doesn't already exist (on first iteration), then create variables. Otherwise, load model. 
+ Important: Clean up 1) while loop, 2) TRAINING boolean, 3) logging file

particles_batch.py
+ see file for further todos


train_model.sh
+ Instead of running the python script multiple times in order to simulate logging, continuously print to the file as my python script is running (just as I would with stdout)
+ Implement a better way to access folders. Currently, we need to be in Tensorflow roo. 
+ Improve script to handle situation where saved file is not present (generate first saved file)
+ Implement better method to interrupt model (kill more gracefully)


Rescources: 
+ Image Processing Tutorial to Explore: http://www.scipy-lectures.org/advanced/image_processing/#displaying-images
+ Article on Visualizing Neural Networks: http://yosinski.com/deepvis
+ Summary of CNN for Visual Recognition (cs231 notes): http://cs231n.github.io/neural-networks-1/
+ Tutorial based overview of convolutional networks (good for possible ideas): http://deeplearning.net/tutorial/lenet.html