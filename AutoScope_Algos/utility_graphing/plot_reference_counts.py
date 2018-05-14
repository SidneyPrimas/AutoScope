import numpy as np
import matplotlib.pyplot as plt


# 10um: Day 1
color = 'blue'
particle_hpf = [
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
	0.186963951
]

# RBC: Day 1
# color = 'red'
# particle_hpf = [
# 	45.61920393, 
# 	25.42709727,
# 	20.19210666,
# 	13.46140444,
# 	8.226413823,
# 	6.730702219,
# 	28.41852048,
# 	23.18352987,
# 	21.68781826,
# 	27.67066468,
# 	23.93138567,
# 	12.71354864
# ]

# WBC: Day 1
# color = 'black'
# particle_hpf = [
# 	3.55231506,
# 	2.617495307,
# 	2.430531357,
# 	4.861062714,
# 	4.487134813,
# 	5.234990615,
# 	2.243567406,
# 	2.804459258
# ]

# 10um: Day 2
# color = 'blue'
# particle_hpf = [
# 	20.19210666,
# 	19.44425086,
# 	19.44425086,
# 	9.348197527,
# 	11.21783703,
# 	4.487134813
# ]

# RBC: Day 1
# color = 'red'
# particle_hpf = [
# 	29.91423209,
# 	20.93996246,
# 	36.6449343,
# 	39.63635751,
# 	32.15779949,
# 	31.40994369,
# 	35.1492227,
# 	18.69639505,
# 	29.91423209
# ]

# WBC: Day 2
# color = 'black'
# particle_hpf = [
# 	2.243567406,
# 	1.121783703,
# 	0,
# 	1.121783703,
# 	1.869639505,
# 	2.804459258,
# 	0.560891852,
# 	1.495711604,
# 	1.308747654,
# 	2.243567406,
# 	0.747855802,
# 	0.747855802
# ]


# Convert to a numpy array
particle_hpf = np.array(particle_hpf)

label = [
	'FoV 01',
	'FoV 02',
	'FoV 03',
	'FoV 04',
	'FoV 05',
	'FoV 06',
	'FoV 07',
	'FoV 08',
	'FoV 09',
	'FoV 10',
	'FoV 11',
	'FoV 12',
	'FoV 13',
	'FoV 14',
]

# Calcualte Statistisc
mean = np.mean(np.array(particle_hpf))
std = np.std(np.array(particle_hpf))
print "Mean: %f"%(mean)
print "Standard Deviation: %f"%(std)

fig = plt.figure()
plt.scatter(label[:particle_hpf.shape[0]], particle_hpf, color=color, s=120, facecolors='none')

# Plot statistics 
mean_list = [mean for _ in range(len(label))]
std_top_list = [11.40480098 for _ in range(len(label))]
std_bottom_list = [0 for _ in range(len(label))]
#plt.plot(label,mean_list, '--', linewidth=1.0, color = 'black')
#plt.plot(label,std_top_list, '--', linewidth=1.0, color = 'red')
#plt.plot(label,std_bottom_list, '--', linewidth=1.0, color = 'red')
plt.xticks(rotation=30, fontsize='12')

# Configure plot
fig.patch.set_facecolor('white')
plt.xlabel("Different Fields of View", fontsize="15")
plt.ylabel("Particles Per HPF", fontsize="15")
plt.show()