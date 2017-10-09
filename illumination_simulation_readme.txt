### Calcualting Intensity from Luminious Flux ###
+ The luminious flux is the total power emitted by the LED. The equation given in the papers is given in terms of luminious intensity, or the power per steradians (or spherical unit area). Since our LED (white LED) has a Lambertian emission pattern, we can make the conversion from luminious flux to luminous intensity through the cosine rule. The details are given here: https://en.wikipedia.org/wiki/Lambert%27s_cosine_law#Relating_peak_luminous_intensity_and_luminous_flux. The result of the calculation is Luminious Flux = pi*Max Luminious Intensity. 
+ Output of Equation is Irradiance Distribution (E): The irradiance distriubton is the power/m^2. The distributoin is highly dependent on z, the distance the light is away from the surface. When the distance between the surface and the LED is 1m, the max irradiance is equal to the max LED intensity (due to the defenition of steradians is r^2). At 1m, the steradian is 1m^2 in area and the irradiance is also 1m. 
+ Determining z (distance from LEDs to surface): The question is if z is the distance to the sample or to the lens. The answer is z is the distance from the LEDs to the lens. And, at the LED, we measure it from the substrate (since that's actually producing the intensity) and not the lens (which is just redirection the beams)

# Rescources
+ LED with built in Lambertian lens: http://www.lumileds.com/uploads/54/DS51-pdf
+ High-powered LED without built-in lens: http://www.mouser.com/ProductDetail/Lumileds/MXA8-PW65-H001/?qs=PVVDbbWpW3Lf3I79tTFqfw%3D%3D
+ Explanation of Flux and Intensity: http://www.jensign.com/LEDIntensity/
+ Explanation of Flux and Intensity: http://www.giangrandi.ch/optics/lmcdcalc/lmcdcalc.shtml



### Storing Old Code ###
+ Change view of 3d matplot lib graph
ax = fig.add_subplot(111, projection='3d')
square_map = ax.plot_surface(X, Y, E, cmap= "gray") 
ax.view_init(elev=90, azim=0)
fig.colorbar(square_surf, orientation="vertical")