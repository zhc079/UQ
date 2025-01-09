import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

data=np.loadtxt("J_id_seg1_seg2_ave_std_sobol.txt")

id=data[:,0]
J=data[:,3]
std=data[:,4]
sobol=data[:,5]
nbins = 50
values=np.vstack([J, std])
k = gaussian_kde(values)
xi, yi = np.mgrid[J.min():J.max():nbins*1j, std.min():std.max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
plt.contour(xi, yi, zi.reshape(xi.shape) )



#plt.yscale("log")
#plt.ylim([1e-20,1e-1])
#plt.scatter(J,std)
plt.show()

