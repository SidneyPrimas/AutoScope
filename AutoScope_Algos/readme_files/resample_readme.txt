#### Downsampling + Resampling Notes ####
+ Results of Resampling: The question with resampling is if information about the image is lost through the resampling process. 
++ Upsample: When an image is upsampled, we do not lose information. Instead, we just spread information across more pixels. Upsampling is similar to increasing the number of pixels when the image is limited by analog resolution of the optics.  
++ Downsample: When an image is downsampled, we do lose information as the higher frequency components start disappearing. Downsampling is like reducing the digital resolution of the image sensor. 

+ The goal: Our goal with downsampling is to make the image appear equivilant to an original image with lower resolution (due to less magnification or lower pixel density). Essentially, we want to convert a higher pixel resolution image (digital resolution) to a lower digital resolution. One assumption is that the original image is limited by digital resolution (the pixels) and not analog resolution (the optics). If it were limited by analog resolution, then downsampling the pixels would not necesarily reduce the information since it still might be better than the analog resolution. So, if we downsample an image that is limited by digital resolution, then we get an image with a lower digital resolution. Thus, the image would be similar to one taken by a lower digitial resolution in the first place. Of course, they would not be exactly the same since the down-sampling process would be different when the resolution is physically different (less pixels in real image sensor). However, we believe that they would be similar. 

+ Maintaining um/pixel resolution: The goal with resampling is to go from um/pixel resolution to antoher. 
++ The Assumption: First, we assume that all the um/pixel resolutions of the images are the same. Essentially, we have a single camera at a single magnification taking all the pictures. Once the images have been taken, each eamage is cropped differently. 
++ The Strategy: Since we want to change the um/pixel, we need to downsample based on the existing number of pixels within each cropped image. If an image has 10x10 pixels at 1pixel/um and we want to go to 0.5pixel/um, then we need to resample to 5x5 pixels. This resampling is dependent on the number of pixels present in the cropped image. Once we have resampled the image to the right um/pixel, we can then upsample the image so that we have a uniform number of pixels that we are entering into the Tensorflow algorithem. The high level goal is to limit the information in the image to only what we would get at x um/pixel. Once we have brought the information down to that level, we can upsample since this will introduce no new information. The key here is that all images must be upsampled (and no images downsampled). So, we need to pick the upsampling based on the image with the largest dimension. 

++ Problem with above approach: This approach creates a huge neural net that takes a long time to train. Ideally, we would like to reduce the size of the images used in the algorithem. This should be fine since the large images include large objects that should be very easy to identify. 


### Implementation Notes ###
+ Notes on Interpolation: By default, we interpolate the base pixels in the image when we use image show. To turn this off, we set the interpolation to 'none'. However, 'none' isn't avaialbe, so we use 'nearest'. The 'nearest' interpolation is equivilant to 'none' at normal scale, but allows for reinterpolation when the image is scaled in a pdf.
+ Downsampling/Upsampling with Bilinear Interpolation: https://math.stackexchange.com/questions/48903/2d-array-downsampling-and-upsampling-using-bilinear-interpolation

#### Execution Notes ####
+ When you run resample.py, make sure you run it from the Tensorflow directory (the root directory). The relative internal paths have been defined assuming the code is run from the root directory. 


### TO DO ### 
+ Instead of overwriting the existing file, we need to keep the original file structure, and save into a new file structure. This way, we keep the original and altered images. The shortcut is to just copy the original files, and then modify the copy. 
+ Downsample: When we resample the image to the same final dimensions (the last step), we sometimes downsample a lage image to 52x52px. However, the resampling algo used is bilinear, which is not good for downsampling. 