from scipy.optimize import curve_fit
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Epithelial Cell
total = 209.0
color = 'blue'
counts = np.array([0, 5, 10, 20], dtype=np.float64)
culm_urine_counts_inv = np.array([209,25, 12, 3], dtype=np.float64)/total
culm_urine_counts = np.array([0,155,191, 203], dtype=np.float64)/total
abs_urine_counts = [155, 36, 12, 6]

# WBC
total = 209.0
color = 'black'
counts = np.array([0, 5, 10, 20], dtype=np.float64)
culm_urine_counts_inv = np.array([209, 94, 52, 30], dtype=np.float64)/total
culm_urine_counts = np.array([0,115, 157,179], dtype=np.float64)/total
abs_urine_counts = [115, 42, 22, 30]

# RBC
# color= 'red'
# total = 209.0
# counts = np.array([0, 5, 10, 20], dtype=np.float64)
# culm_urine_counts_inv = np.array([209, 29, 22, 14], dtype=np.float64)/total
# culm_urine_counts = np.array([0,180, 187,195], dtype=np.float64)/total
# abs_urine_counts = [180, 7, 8, 14]

# Fit 2 exponential curve to culmative urine counts
optimal_params, _ = curve_fit(lambda t,a1,b1, a2, b2: a1*np.exp(b1*t)+a2*np.exp(b2*t)+1,  counts,  culm_urine_counts, p0=[-1,-1,-1,-1], method='lm') 
print optimal_params
a1, b1, a2, b2 = optimal_params

# Obtain CDF an dpDF
count = np.array(np.linspace(0,100,1000))
cdf = a1*np.exp(b1*count) + a2*np.exp(b2*count) + 1
pdf = b1*a1*np.exp(b1*count) + b2*a2*np.exp(b2*count)

# Sanity check that 
pdf_sum =  np.trapz(pdf[0:50], count[0:50])
print "Sum of Derivative: %f"%(pdf_sum*total)
# if (pdf_sum > 1.1) or (pdf_sum < 0.85):
# 	print pdf_sum
# 	raise ValueError("The PDF doesn't sum to a number that is sufficiently close to 1.")

# Generate the urine counts from the distribution. 
# Eventhough the PDF integrates to 1. We need deriv_y to sum to 1 (not same as integral).
selected_counts = np.random.choice(count,size=10000, replace=True, p=pdf/sum(pdf))

# Plot results
fig = plt.figure()
plt.plot(count, cdf, ':', color=color)
plt.scatter(counts, culm_urine_counts, color=color)
plt.xlabel("Particle Counts (Per HPF)", fontsize="15")
plt.ylabel("Cumulative Probability", fontsize="15")
axes = fig.axes[0]
axes.set_xlim([0,30])
axes.set_ylim([0,1])

fig = plt.figure()
plt.plot(count, pdf, color=color)
n, bins, patches = plt.hist(selected_counts, bins=100, normed=True, facecolor='green', alpha=0.75)
plt.xlabel("Particle Counts (Per HPF)", fontsize="15")
plt.ylabel("Probability", fontsize="15")
axes = fig.axes[0]
axes.set_xlim([0,30])
axes.set_ylim([0,1])
plt.show()