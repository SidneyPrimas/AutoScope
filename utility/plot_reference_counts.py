import numpy as np
import matplotlib.pyplot as plt




particle_10um_hpf = [
	8.226413823,
	5.982846417,
	0.747855802,
	3.36535111,
	11.40480098,
	2.617495307,
	1.121783703,
	0.747855802,
	1.121783703,
	0,
	5.608918516,
	7.665521972,
	0.186963951,
	3.753660853,
]

# Convert to a numpy array
particle_10um_hpf = np.array(particle_10um_hpf)

label = [
	'View 01',
	'View 02',
	'View 03',
	'View 04',
	'View 05',
	'View 06',
	'View 07',
	'View 08',
	'View 09',
	'View 10',
	'View 11',
	'View 12',
	'View 13',
	'View 14',
]

# Calcualte Statistisc
mean = np.mean(np.array(particle_10um_hpf))
std = np.std(np.array(particle_10um_hpf))
print "Mean: %f"%(mean)
print "Standard Deviation: %f"%(std)

fig = plt.figure()
plt.plot(label, particle_10um_hpf, 'b.', markersize='20')

# Plot statistics 
mean_list = [mean for _ in range(len(label))]
std_top_list = [11.40480098 for _ in range(len(label))]
std_bottom_list = [0 for _ in range(len(label))]
#plt.plot(label,mean_list, '--', linewidth=1.0, color = 'black')
#plt.plot(label,std_top_list, '--', linewidth=1.0, color = 'red')
#plt.plot(label,std_bottom_list, '--', linewidth=1.0, color = 'red')
plt.xticks(rotation=30)

# Configure plot
fig.patch.set_facecolor('white')
plt.xlabel("Different View Positions on Slide", fontsize="20")
plt.ylabel("Microbeads Per HPF", fontsize="20")
plt.show()