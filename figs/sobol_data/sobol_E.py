import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

data=np.loadtxt("Eid_ave_std_sobol.txt")

id=data[:,0]
E=data[:,1]
std=data[:,2]
sobol=data[:,3]
nbins = 50
values=np.vstack([E, sobol])
k = gaussian_kde(values)
xi, yi = np.mgrid[E.min():E.max():nbins*1j, sobol.min():sobol.max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
plt.contour(xi, yi, zi.reshape(xi.shape) )
#plt.yscale("log")

plt.scatter(E,std)
plt.show()

