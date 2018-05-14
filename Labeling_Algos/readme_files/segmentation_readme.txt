+ Watershed Theory: Select markers (either manually or algorithmically). These markers represent local elevation. The watershed method floods valleys, using the markers as the sources of the water. The flood continues until the sources of water start overlapping at the ridges of the particles. Whenever there is an overlap, a "wall" is built, and the flooding continues until the next overlap. The results walls are the segmentation.
+ Watershed Implementation: However, usuing raw pixel intensities is too noisy. Instead, we label the images into 3 sections: definitely foreground, definitely background and unsure.  
+ Adaptive thresholding (written for Microscope images): When we have varying illumination across the image, use adaptive threshold. Here, the threshold is calculated for a small region of the image, and then applied to that region. This is done individually for each and every pixel. 
++ The threshold is determined based on the mean of pixels in block-size minus C constant. If a pixel is above that threshold, we set it to black.  Thus, if have a higher constant, the threshold gets lower, and we return more 255s (or white). So, high constants eliminate the noise, but also removes the particles. The benefit of high block sizes is that it makes sure that we capture the edge of a particle. If we don't, the center of the particle might be white (255) since it sets the threshold based only on the particle. 
+ Otsu's Binarization with Adaptive Filtering: When the pixel distribution is bimodal, Otsu's method selects a threshold based on the histogram's bimodal distribution. In this case, set thresh = 0, and add cv2.THRESH_OTSU as an option. The return value is the thresh selected.
+ Adaptive Thresholding (for Reversed Lens): In the inverse case, when a pixel is below the threshold, we set the pixel to white (255). Otherwise, we set it to black (0). Since the particles are very white, any pixel that includes the particle as part of it's block will have a high threshold. Thus, a particle in the background will fall below the threshold, and thus will be set to white. However, this only happens at the edges of the particle. 
++ Small vs. Large Blocks: With large blocks, the center of the particle will be set to black. Essentially, the large blocks will capture the center of the particle as well as the background. Since the background is captured as well, these center pixels will be higher than the threshold and thus will be black. With small blocks: 1) the edges will still be white since the small block straddles the particle and 2) the center of the particle will be more random (instead of just black) since the center will be determined based on a threshold of only the particles. 
++ Big vs Small Constant: The constant is an indicator of how sharp the contrast needs to be from it's surrounding edges. The higher the constant, the more the results will be biased to either black and white, and the less the slight contrasts will be detected. 


+ Dilation: Usually used for binary images. Enlarges the foreground (usually white pixels). The kerenl is centered on each background pixel. For every background pixel, we place the origin of the kernel on it. If any part of the kernel includes the foreground, then the target/input pixel is flipped from background to foreground. Thus, the foreground is enlarged. 
+ Erosion: Usually used on binary images. Reduces the size of the foreground (usually the white pixels). In this iteration, we center the kernel on each foreground pixel. If any part of the kernel is part of the background, then change the target pixel from foreground to background thus decreasing the foreground. 
+ Closing (removes dots from foreground): Dilation followed by erosion. Once a closing is applied (with a single iteration of erosion and dislation) then there will be no more changes when more closings are applied. Visual rule of thumb: If the kernel cannot be placed to include a background without including any foreground then that point must be flipped to foreground. So, we switch background to foreground. 
+ Opening (removes dots from background): Similar to erosion (it reduces the foreground), but less desctructive. Opening preserves the foreground that has a similar shape to the kernel or is contained by the kernel. If that isn't the case, it reduces the foreground, eroding a few pixels. This eliminates any pop-corn noise. Visual rule of thumb: If the kernel cannot be placed to include the foreground without including the background then those pixels must be switched to background. 

distanceTransform: Caculates the distance from every pixel to the nearest 0 pixel (the nearest background pixel). This distance is calculated using different pixel shifts,including: vertical/horizontal shift, diagnoal shift, and horse's shift. Each of these costs a different value in the distance calculation, thus giving you a total dinstance for each pixel. In the C version, you can also labels for different parts of the figure. 

Design Decision: 
+ Group Clumps: Ideally we group clumps. By seperating clumps, we would just recount them multiple times in the different 52x52 images. Instead, we want a single 52x52 image for a clump. To group clumps, we need to dilate the images so that clumps actually expand to touch each other. Then, each clump will be a continious area instead of discreate components. If we did not want to group clumps, then we would find local maxima for each partical (based on the distance formula). And, we would use these maxima to segment the particles. 

Convention: 
+ 8bit visuals: 255 is white and 0 is black. 
+ Foreground vs Bacground: Visually, the foreground is white while the background is white. What are the numbers associated with the foreground and background? 

ToDo: 
+ Create 'File Meta Data Summary': The file should include 1) file locations, 2) and meta-data used to do segmentation. 
+ Get states from connectedComponents() functions. Use the stats to eliminate components that are obviously not cells. 
+ In the future: Do more detailed component analysis on all the identified components. In fact, think about doing an Independent Component Analysis or Principle Component Analysis. 
+ Look into Image Denoising: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html
+ Need to improve segmentation for reversed lens approach: 1) use a better illumination compensation, 2) only focus on certain areas of image, 3) improve detection algo, etc. 


Rescources: 
+ Watershed Tutorial: http://www.pyimagesearch.com/2015/11/02/watershed-opencv/
+ Watershed Tutorial: http://docs.opencv.org/3.1.0/d3/db4/tutorial_py_watershed.html 
+ Image Thresholding: http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html
+ Description of opening/closing/erosion/dilation: http://homepages.inf.ed.ac.uk/rbf/HIPR2/open.htm#guidelines
+ Image Denoising Tutorial: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html


