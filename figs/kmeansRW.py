from sklearn.cluster import KMeans
from sklearn.neighbors import RadiusNeighborsClassifier
import numpy as np
import math
import matplotlib.pyplot as plt

N = 1000

x1 = np.loadtxt("PBE0/eigenvec1RW.txt")
x2 = np.loadtxt("PBE/eigenvec1RW.txt")
x3 = np.loadtxt("B3LYP/eigenvec1RW.txt")
x4 = np.loadtxt("BHANDHLYP/eigenvec1RW.txt")
x5 = np.loadtxt("TPSS/eigenvec1RW.txt")
x6 = np.loadtxt("BP86/eigenvec1RW.txt")
x7 = np.loadtxt("wB97X/eigenvec1RW.txt")
x8 = np.loadtxt("wB97X-D3/eigenvec1RW.txt")
x9 = np.loadtxt("M06L/eigenvec1RW.txt")
x10 = np.loadtxt("BHLYP/eigenvec1RW.txt")
ToF = [1.23e-3,1.04e-3,4.28e-3,737.94,1.37e-2,7.46e-3,2.92e-2,0.0489,0.319,2.47]

dict_eigVec = {1:x1,2:x2,3:x3,4:x4,5:x5,6:x6,7:x7,8:x8,9:x9,10:x10}
dict_func = {1:"PBE0",2:"PBE",3:"B3LYP",4:"BHAND",5:"TPSS",6:"BP86",7:"wB97X",8:"wB97X-D3",9:"M06L",10:"BHLYP"}


fig,ax = plt.subplots(figsize=(10, 6))
for i in np.arange(1,11,1):
    ax.scatter(np.ones(N)+i-1,dict_eigVec[i],label=dict_func[i])
ax.set_xlim(0,15)
ax.tick_params(axis='y', labelsize=16)

ax.set_ylabel(r"eigenvector elements",fontsize=18)
ax.legend()
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) 
fig.savefig("fig_distri_eigenVecRW.pdf",dpi=300)
##################################### get cost kMeans for 2c ######################
all_cost = []

for i in np.arange(1,11,1):
    data = np.zeros((N,2))
    data[:,0] = dict_eigVec[i]
    data[:,1] = 1
    kmeans = KMeans(n_clusters=2).fit(data)
    cluster0 = np.where(kmeans.labels_ == 0)[0]
    cluster1 = np.where(kmeans.labels_ == 1)[0] # sth like array[66, 738]
    center0 = kmeans.cluster_centers_[0][0]
    center1 = kmeans.cluster_centers_[1][0]
    cost = 0
    for ii in range(cluster0.shape[0]):
        cost += (data[cluster0[ii],0] - center0 )**2
    for ii in range(cluster1.shape[0]):
        cost += (data[cluster1[ii],0] - center1 )**2
    all_cost.append(cost)
    print('--info: cost is:',cost)
    print("--info: cluster size:", len(cluster0),len(cluster1))

all_cost_3c = []
for i in np.arange(1,11,1):
    data = np.zeros((N,2))
    data[:,0] = dict_eigVec[i]
    data[:,1] = 1
    kmeans = KMeans(n_clusters=3).fit(data)

    cluster0 = np.where(kmeans.labels_ == 0)[0]
    cluster1 = np.where(kmeans.labels_ == 1)[0] # sth like array[66, 738]
    cluster2 = np.where(kmeans.labels_ == 2)[0]
    center0 = kmeans.cluster_centers_[0][0]
    center1 = kmeans.cluster_centers_[1][0]
    center2 = kmeans.cluster_centers_[2][0]
    cost = 0
    for ii in range(cluster0.shape[0]):
        cost += (data[cluster0[ii],0] - center0 )**2
    for ii in range(cluster1.shape[0]):
        cost += (data[cluster1[ii],0] - center1 )**2
    for ii in range(cluster2.shape[0]):
        cost += (data[cluster2[ii],0] - center2 )**2
    print('--info: cost is:',cost)
    all_cost_3c.append(cost)

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10, 5))
ax1.scatter(ToF,all_cost)
#ax.set_xlim(0,10)
ax1.set_xlabel(r"$ToF [s]$",fontsize=16)
ax1.set_ylabel(r"$Z_{2c}$",fontsize=16)
#ax.legend()
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.tick_params(axis='both', labelsize=16)

ax2.scatter(ToF,all_cost_3c)
#ax.set_xlim(0,10)
ax2.set_xlabel(r"$ToF [s]$",fontsize=16)
ax2.set_ylabel(r"$Z_{3c}$",fontsize=16)
#ax.legend()
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.tick_params(axis='both', labelsize=16)

plt.tight_layout()
fig.savefig("fig_Z_ToF_RW.pdf",dpi=400)