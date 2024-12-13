import matplotlib.pyplot as plt
import scipy.stats as sps
import numpy as np

x_n = sps.norm.rvs(loc=0, scale=1, size=5)
x_c = sps.cauchy.rvs(loc=0, scale=1, size=5)
print("x_n", x_n)
print("x_n", np.sort(x_n))
print()
print("x_c", x_c)
print("x_c", np.sort(x_c))
print("med_n = ", np.median(x_n))
print("med_c = ", np.median(x_c))
linsp = [i for i in range(0, 100)]
plt.hist(x_n, bins=30, density=True)
plt.hist(x_c,  bins=100, alpha=0.5, density=True)
plt.show()
