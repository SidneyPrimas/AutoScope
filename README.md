# The AutoScope: An Automated, Point-of-Care Urinalysis System 
**Advisor:** [Prof. Charlie Sodini](http://imes.mit.edu/people/faculty/sodini-charles/), LeBel Professor of Electrical Engineering, MIT  
**Sponsorship:** [Medical Electronic Device Realization Center (MEDRC)](http://medrc.mit.edu/) - [Analog Devices, Inc.](https://en.wikipedia.org/wiki/Analog_Devices) 

I spent 2 years working on my Master's in Computer Science at MIT. I developed my own low-cost microscope (the Autoscope) and used neural networks to automatically classify particles in urine. My work enables doctors to do low-cost urinalysis at the point-of-care instead of sending it off to a laboratory and waiting a few days for the results.   
My low-cost microscope does not have any magnification and so it *shouldn't* be possible to detect red blood cells. But the cool part is that... it does.   
This work highlights the power of neural networks to take advantage of information that we, as humans, cannot.   


**Video of Final Project Presentation (28min):**   
**Slides of Final Project Presentation:** put them on slideshare  
**Master's Thesis:** pdf  
___
### Abstract  
Over 200 million urine tests are ordered each year in the US alone. Due to the cost and complexity of microscopic urinalysis tests, the majority are conducted at a central medical lab instead of the point-of-care. The AutoScope is an automated, low-cost microscopic urinalysis system that can accurately quantify red blood cells (RBCs) and white blood cells (WBCs) at the point-of-care. Even without any magnification, we achieved sensitivity, specificity, and R-squared values that are comparable (and mostly better) than the same metrics for the iQ-200, a $100,000-$150,000 state-of-the-art semi-automated urinalysis system. Specifically, the AutoScope’s particle counts and the reference particle counts (cross-validated through medical laboratory results) were linearly correlated to each other (r2= 0.980) for RBCs and WBCs. Furthermore, the AutoScope has an estimated sensitivity of 88% (RBCs) and 91% (WBCs) and an estimated specificity of 89% (RBCs) and 97% (WBCs). 

  
### Description of Code Base
My code is written in Python and is organized into 3 folders: 

##### 1. AutoScope_Algos
This folder contains the machine learning algorithms for classification and segmentation of particles in the Autoscope's images. It contains 3 sub-folders:
* **core_algo** - this sub-folder contains the workhorse functions. 
  * **data_preparation** - these scripts put my microscope's raw data into the proper formats and folder structure for analysis   
  The **most important** scripts are: 
    * Scripts that train my neural network on both particle segmentation and classification: **train_classification_particles.py** and  **train_segment_particles.py**   
    * Scripts that perform both particle segmentation and classification: **process_urine_classify.py**	and **process_urine_segment.py** 
* **utility_functions** - supporting functions
* **utility_graphing** - supporting functions
  
##### 2. Labeling_Algos
These scripts build tools that allow a user to manually label particles in Autoscope images in order to develop a training dataset. The training datasets are fed into the "train_*_particles.py" scripts above. 

##### 3. Sub_Tasks
These scripts perform other types of analyses needed for my Master's thesis that are not related to classification or segmentation of particles. 
* **20170810_lighting** - these scripts model the illumination pattern of the Autoscope
* **20171030_resolution_exp** - these scripts calculate the end-to-end resolution of the Autoscope system



--- SCRAPS ---
that models the illumination pattern of the Autoscope, calculating the end-to-end resolution of my microscope 

  
  Training scripts: 
  train_classification_particles.py	Reorganize fildes/folders to easier navigate project.	3 days ago
train_segment_particles.py

Processing scripts: 
process_urine_classify.py	Reorganize fildes/folders to easier navigate project.	3 days ago
process_urine_segment.py

Prepares data to be processed (right folder structure from raw input data) 
used to train classification models, train the segmentation models, and predict the cell counts in the Autoscope's images


