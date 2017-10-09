"""
    File name: illumination_simulation.py
    Author: Sidney Primas
    Date created: 08/09/2017
    Python Version: 2.7
    Description: Simulates the illumination pattern of an specific LED. 
"""
# Base packages 
import numpy as np
import matplotlib.pyplot as plt
import math

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

# Defined System Parameters
# Note: The units of distance is in meter. 
m = 1.0 # Used to estimate the angular intensity distribution 
# Datasheet only provides luminous flux (total power emitted by LED)
lum_flux = 45e-3 # Typyical Flux (in W). Depends on current. 
surface_size =20e-3
led_height = 20e-3 # use 15mm
roi_size = 10e-3

# Calculate Max Luminious Intensity for Lambertian
I_led_max = lum_flux/math.pi # LED intensity in W/sr. See readme. 

# Define Coordinates (in mm)
x = np.linspace(-surface_size/2, surface_size/2, 30)
y = np.linspace(-surface_size/2,surface_size/2, 30)
X, Y = np.meshgrid(x, y)
z = led_height # Distance between LED and detector 

# Define Equation for single LED (where LED is at (0,0))
X_loc = 0 # X Location of LED
Y_loc = 0 # Y Location of LED
base = (np.power(X-X_loc,2) + np.power(Y-Y_loc,2) + np.power(z,2))
E = math.pow(z,m) * I_led_max * np.power(base,-(m+2)/2)


fig = plt.figure()
ax = fig.add_subplot(111)
single_map = plt.pcolormesh(X, Y, E, cmap= "gray", shading='gouraud') 
fig.colorbar(single_map, orientation="vertical", label='Intensity Distribution (W/m^2)')
plt.title('Single LED Configuration')
plt.xlabel('X distance from Lens(in meters)')
plt.xlabel('Y distance from lens (in meters)')


# Define Equation for two LEDs
d_max = math.sqrt(4.0/(m+3))*z #The ideal distance between LEDs is the height of the LEDs with a Lambertian 
print "Two LEDs d_max: %f"%(d_max)
d = d_max # Distance between LEDs
base1 = (np.power(X-d/2,2) + np.power(Y,2) + np.power(z,2))
base2 = (np.power(X+d/2,2) + np.power(Y,2) + np.power(z,2))
E = math.pow(z,m) * I_led_max * (np.power(base1,-(m+2)/2) + np.power(base2,-(m+2)/2))

fig = plt.figure()
ax = fig.add_subplot(111)
double_map = plt.pcolormesh(X, Y, E, cmap= "gray", shading='gouraud') 
fig.colorbar(double_map, orientation="vertical", label='Intensity Distribution (W/m^2)')
plt.title('Double LED Configuration')
plt.xlabel('X distance from Lens(in meters)')
plt.xlabel('Y distance from lens (in meters)')





# Define Equation for 4 LEDs (N and M are even numbers)
d_max = math.sqrt(4.0/(m+2))*z # For 4 LEDs, the distnace increases since there is more overlap of the different light beams. 
print "4 LEDs d_max: %f"%(d_max)
d = d_max # Distance between LEDs
base1 = (np.power(X+d/2,2) + np.power(Y+d/2,2) + np.power(z,2))
base2 = (np.power(X+d/2,2) + np.power(Y-d/2,2) + np.power(z,2))
base3 = (np.power(X-d/2,2) + np.power(Y+d/2,2) + np.power(z,2))
base4 = (np.power(X-d/2,2) + np.power(Y-d/2,2) + np.power(z,2))
E = math.pow(z,m) * I_led_max * (np.power(base1,-(m+2)/2) +  
	np.power(base2,-(m+2)/2) + 
	np.power(base3,-(m+2)/2) + 
	np.power(base4,-(m+2)/2))



fig = plt.figure()
ax = fig.add_subplot(111)
# Use pcolormesh since it's faster than pcolor
# Use shading for better visualization. The other options are: 1) interpelation built into imshow and 2) interpolation using interp2d from scipy
square_map = plt.pcolormesh(X, Y, E, cmap= "gray", shading='gouraud') 

fig.colorbar(square_map, orientation="vertical", label='Intensity Distribution (W/m^2)')
plt.title('Square LED Configuration')
plt.xlabel('X distance from Lens(in meters)')
plt.xlabel('Y distance from lens (in meters)')

# Indicate the Rregion of Interest
ax.add_patch(
    patches.Rectangle(
        (-roi_size/2, -roi_size/2),   # (x,y)
        roi_size,          	# width
        roi_size,			# height
        fill=False			# remove background
    )
)

# Indicate LED Location
ax.add_patch(patches.Circle((d/2,d/2), 0.0001, fill=False))
ax.add_patch(patches.Circle((d/2,-d/2), 0.0001, fill=False))
ax.add_patch(patches.Circle((-d/2,d/2), 0.0001, fill=False))
ax.add_patch(patches.Circle((-d/2,-d/2), 0.0001, fill=False))




plt.show()


