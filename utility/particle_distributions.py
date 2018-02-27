from scipy.optimize import curve_fit
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Epithelial Cell
title = "Epithelial Cells"
total = 209.0
color = 'blue'
counts = np.array([0, 5, 10, 20], dtype=np.float64)
culm_urine_counts = np.array([209,25, 12, 3], dtype=np.float64)/total
# abs_urine_counts = [184, 13, 9, 3]

# WBC
title = "White Blood Cells"
total = 208.0
color = 'black'
counts = np.array([0, 5, 10, 20], dtype=np.float64)
culm_urine_counts = np.array([208, 79, 57, 41], dtype=np.float64)/total
# abs_urine_counts = [129, 22, 16, 41]

# RBC
title= "Red Blood Cells"
color= 'red'
total = 209.0
counts = np.array([0, 5, 10, 20], dtype=np.float64)
culm_urine_counts = np.array([208, 29, 22, 14], dtype=np.float64)/total
# abs_urine_counts = [180, 7, 8, 14]

# Fit 2 exponential curve to culmative urine counts
optimal_params, _ = curve_fit(lambda t,a1,b1, a2, b2: a1*np.exp(b1*t)+a2*np.exp(b2*t),  counts,  culm_urine_counts, p0=[1,-1,1,-1], method='lm') 
print optimal_params
a1, b1, a2, b2 = optimal_params

# Obtain CDF an dpDF
count = np.array(np.linspace(0,50,1000))
cdf = a1*np.exp(b1*count) + a2*np.exp(b2*count)
pdf = -b1*a1*np.exp(b1*count) + -b2*a2*np.exp(b2*count)

# Sanity check that 
pdf_sum =  np.trapz(pdf, count)
print "Sum of Derivative: %f"%(pdf_sum)
if (pdf_sum > 1.1) or (pdf_sum < 0.9):
	raise ValueError("The PDF doesn't sum to a number that is sufficiently close to 1.")

# Generate the urine counts from the distribution. 
# Eventhough the PDF integrates to 1. We need deriv_y to sum to 1 (not same as integral).
selected_counts = np.random.choice(count,size=10000, replace=True, p=pdf/sum(pdf))

# Plot results
fig = plt.figure()
plt.plot(count, cdf, ':', color=color)
plt.plot(count, pdf, color=color)
plt.scatter(counts, culm_urine_counts, color=color)
n, bins, patches = plt.hist(selected_counts, bins=100, normed=True, facecolor='green', alpha=0.75)
axes = fig.axes[0]
axes.set_xlim([0,30])
plt.show()