"""
    File name: plot_resolution.py
    Author: Sidney Primas
    Date created: 10/30/2017
    Python Version: 2.7
    Description: Function to plot resolution results
  	Notes: We took 5 resolution images. We through out 1 image (img3 at 432px) since it didn't match with the actual results. 
  	The rest are approximatinos of visual and resolution.py analysis. The major issue is that the target is not straight. 
  	Since the target isn't straight, our analysis isn't exactly accurate. 
"""


# Base packages 
import numpy as np
import matplotlib.pyplot as plt
pixel_size = 1.42

x = [43.0, 694.0, 924.0, 1171.0]
x = [pixel_size * pt for pt in x]
y = [5.86, 6.2, 6.58, 8.29] # in um


fig = plt.figure()
fig.patch.set_facecolor('white')
plt.plot(x,y,'ko')
plt.plot(x,y,'k')
plt.xlabel("Distance from Center (um)")
plt.ylabel("Resolution (um)")


plt.show()