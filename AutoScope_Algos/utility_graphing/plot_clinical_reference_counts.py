import numpy as np
import matplotlib.pyplot as plt

# AutoScope Values
rbc_values = [
	[63.88, [33.3, 66.7]], # Soluton 5
	[80.49, [83.3, 95.2]], # Solution 6
	[86.49, [76.9,	93.8]], # Solution 8
]
wbc_values = [
	[34.03, [33.3, 66.7]], # Soluton 5
	[18.02, [4.8, 16.7]], # Solution 6
	[12.84, [6.3, 23.1]], # Solution 8
]

# Reference Values
rbc_values = [
	[60, [33.3, 66.7]], # Soluton 5
	[86, [83.3, 95.2]], # Solution 6
	[90, [76.9,	93.8]], # Solution 8
]
wbc_values = [
	[40, [33.3, 66.7]], # Soluton 5
	[14, [4.8, 16.7]], # Solution 6
	[10, [6.3, 23.1]], # Solution 8
]







fig = plt.figure()
# Plot the unity line
x_percent = range(100)
y_percent = range(100)
identity_plt, = plt.plot(x_percent, y_percent, linewidth=1.0, color='gray')

for particle_results in rbc_values:
	y_label = [particle_results[0] for _ in range(2)]
	rbc_plot, = plt.plot(particle_results[1], y_label, 'r:.', markersize='20')


for particle_results in wbc_values:
	y_label = [particle_results[0] for _ in range(2)]
	wbc_plot, = plt.plot(particle_results[1], y_label, ':.', color='black', markersize='20', mfc='none')


# Configure plot
fig.patch.set_facecolor('white')
plt.legend([identity_plt, rbc_plot, wbc_plot], ['Identity Line', 'RBC', 'WBC'], loc='lower right', prop={'size':12}, frameon=False)
plt.xlabel("Medical Lab Results  (%)", fontsize="20")
plt.ylabel("Reference  Results (%)", fontsize="20")
plt.show()