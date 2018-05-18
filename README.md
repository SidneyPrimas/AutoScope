# The AutoScope: An Automated, Point-of-Care Urinalysis System 
**Advisor:** [Prof. Charlie Sodini](http://imes.mit.edu/people/faculty/sodini-charles/), LeBel Professor of Electrical Engineering, MIT  
**Sponsorship:** [Medical Electronic Device Realization Center (MEDRC)](http://medrc.mit.edu/) - [Analog Devices, Inc.](https://en.wikipedia.org/wiki/Analog_Devices) 
  
  
I spent 2 years working on my Master's in Computer Science at MIT. I developed my own low-cost microscope (the Autoscope) and used neural networks to automatically classify particles in urine. My work enables doctors to do low-cost urinalysis at the point-of-care instead of sending it off to a laboratory and waiting a few days for the results.   
My low-cost microscope does not have any magnification and so it *shouldn't* be possible to detect red blood cells. But the cool part is that... it does.   
This work highlights the power of neural networks to take advantage of information that we, as humans, cannot.   
  
  
  
**Video of Final Project Presentation (26min):** Available on [YouTube](https://youtu.be/SKFaWKCmoxo)  
**Slides of Final Project Presentation:** Available on [Slideshare](https://www.slideshare.net/SidneyPrimas/the-autoscope-an-automated-pointofcare-urinalysis-system)  
**Master's Thesis:** Available [here](https://github.com/SidneyPrimas/AutoScope/blob/master/MIT_Master_Thesis.pdf)  
  
  
  
<img src="https://github.com/SidneyPrimas/AutoScope/blob/master/AutoScope_cover_image.jpeg" alt="CoverImage" width="500">
  
## Abstract  
Over 200 million urine tests are ordered each year in the US alone. Due to the cost and complexity of microscopic urinalysis tests, the majority are conducted at a central medical lab instead of the point-of-care. The AutoScope is an automated, low-cost microscopic urinalysis system that can accurately quantify red blood cells (RBCs) and white blood cells (WBCs) at the point-of-care. Even without any magnification, we achieved sensitivity, specificity, and R-squared values that are comparable (and mostly better) than the same metrics for the iQ-200, a $100,000-$150,000 state-of-the-art semi-automated urinalysis system. Specifically, the AutoScope’s particle counts and the reference particle counts (cross-validated through medical laboratory results) were linearly correlated to each other (r2= 0.980) for RBCs and WBCs. Furthermore, the AutoScope has an estimated sensitivity of 88% (RBCs) and 91% (WBCs) and an estimated specificity of 89% (RBCs) and 97% (WBCs). 

  
## Description of Code Base

### Core Tools
* Python: numpy, matplotlib, cv2, etc
* Tensorflow: Intially, I directly implemented machine learning with Tensorflow. Later, I switched to Keras. 
* AWS: Training of models was done on AWS GPUs


### Code Organization
#### 1. AutoScope_Algos
This folder contains the algorithms for classification and segmentation AutoScope images. The most important scripts are contained in the core_algo directory. Within this directory, I want to highlight a few workhorse scripts that are starting points for different processes. 
 * **Data Preperation Scripts** - These scripts put the Autoscope's images into the proper folder structure necessary for model training. They include data_preperation/create_classification_folder_from_labels.py and data_preperation/create_segmentation_folder.py. 
 * **Training Scripts:** Scripts that train neural networks to perform particle segmentation and classification. These include train_classification_particles.py and train_segment_particles.py. 
 * **Prediction Scripts:** Scripts that used the trained models to predict the particle segmentation and classification on new AutoScope images. These include: process_urine_classify.py	and process_urine_segment.py. 

#### 2. Labeling_Algos
This folder contains scripts to build tools that allow a user to manually label the location and type of particle in Autoscope images. This is done to develop a training dataset. The training datasets are fed into the training scripts above. 

#### 3. Sub_Tasks
This folder contains scripts to perform other types of analyses needed for my Master's thesis that are not related to classification or segmentation of particles. For example, modeling the illumination pattern of the Autoscope, calculating the end-to-end resolution of the Autoscope system, etc.

