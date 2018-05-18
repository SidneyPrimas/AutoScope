# The AutoScope: An Automated, Point-of-Care Urinalysis System 
**Advisor:** [Prof. Charlie Sodini](http://imes.mit.edu/people/faculty/sodini-charles/), LeBel Professor of Electrical Engineering, MIT  
**Sponsorship:** [Medical Electronic Device Realization Center (MEDRC)](http://medrc.mit.edu/) - [Analog Devices, Inc.](https://en.wikipedia.org/wiki/Analog_Devices) 
  
  
I spent 2 years working on my Master's in Computer Science at MIT. I developed my own low-cost microscope (the Autoscope) and used neural networks to automatically classify particles in urine. My work enables doctors to do low-cost urinalysis at the point-of-care instead of sending it off to a laboratory and waiting a few days for the results.   
My low-cost microscope does not have any magnification and so it *shouldn't* be possible to detect red blood cells. But the cool part is that... it does.   
This work highlights the power of neural networks to take advantage of information that we, as humans, cannot.   
  
  
  
**Video of Final Project Presentation (28min):** Available on [YouTube](https://youtu.be/SKFaWKCmoxo)  
**Slides of Final Project Presentation:** Available on [Slideshare](https://www.slideshare.net/SidneyPrimas/the-autoscope-an-automated-pointofcare-urinalysis-system)  
**Master's Thesis:** Available [here](https://github.com/SidneyPrimas/AutoScope/blob/master/MIT_Master_Thesis.pdf)  
  
  
  
<img src="https://github.com/SidneyPrimas/AutoScope/blob/master/AutoScope_cover_image.jpeg" alt="CoverImage" width="500">
  
## Abstract  
Over 200 million urine tests are ordered each year in the US alone. Due to the cost and complexity of microscopic urinalysis tests, the majority are conducted at a central medical lab instead of the point-of-care. The AutoScope is an automated, low-cost microscopic urinalysis system that can accurately quantify red blood cells (RBCs) and white blood cells (WBCs) at the point-of-care. Even without any magnification, we achieved sensitivity, specificity, and R-squared values that are comparable (and mostly better) than the same metrics for the iQ-200, a $100,000-$150,000 state-of-the-art semi-automated urinalysis system. Specifically, the AutoScope’s particle counts and the reference particle counts (cross-validated through medical laboratory results) were linearly correlated to each other (r2= 0.980) for RBCs and WBCs. Furthermore, the AutoScope has an estimated sensitivity of 88% (RBCs) and 91% (WBCs) and an estimated specificity of 89% (RBCs) and 97% (WBCs). 

  
## Description of Code Base
My code is organized into 3 folders: 

#### 1. AutoScope_Algos
This folder contains the algorithms for classification and segmentation AutoScope images. The most important scripts are contained in the core_algo directory. 
* The **most impmortant** scripts are: 
  * Scripts that train neural networks to perform particle segmentation and classification: <strong>train_classification_particles.py</strong> and  <strong>train_segment_particles.py</strong>  
  * Scripts that perform the particle segmentation and classification on new AutoScope images: <strong>process_urine_classify.py</strong>	and <strong>process_urine_segment.py</strong>
 * strong>data_preparation</strong> - these scripts put the Autoscope's images into the proper folder structure necessary for model training   

#### 2. Labeling_Algos
These scripts build tools that allow a user to manually label the location and type of particle in Autoscope images. This is done to develop a training dataset. The training datasets are fed into the "train_*_particles.py" scripts above. 

#### 3. Sub_Tasks
These scripts perform other types of analyses needed for my Master's thesis that are not related to classification or segmentation of particles. For example, modeling the illumination pattern of the Autoscope, calculating the end-to-end resolution of the Autoscope system, etc.

