from sklearn.cluster import KMeans
from sklearn.neighbors import RadiusNeighborsClassifier
import numpy as np
import math
import matplotlib.pyplot as plt

N = 1000

def kde(data):
    h = 1e-5
    input = np.arange(-2,2,0.002)
    kde_result = []
    for i in range(2000):
        temp = 0
        for j in range(data.shape[0]):
            temp += 1/ (h * np.sqrt(2 * math.pi)) * np.exp(-(input[i] - data[j])**2/(2*h))
        kde_result.append(temp/N)
    return input, kde_result

fig,ax = plt.subplots(2,3,figsize=(15,10))
x,y = kde(np.loadtxt("PBE0/eigenvec1.txt"))
ax[0][0].plot(x,y,label="PBE0",color="black",linewidth=3.0)
ax[0][0].set_yscale("log")
x,y = kde(np.loadtxt("PBE/eigenvec1.txt"))
ax[0][1].plot(x,y,label="PBE",color="black",linewidth=3.0)
ax[0][1].set_yscale("log")
x,y = kde(np.loadtxt("B3LYP/eigenvec1.txt"))
ax[0][2].plot(x,y,label="B3LYP",color="black",linewidth=3.0)
ax[0][2].set_yscale("log")
x,y = kde(np.loadtxt("BHANDHLYP/eigenvec1.txt"))
ax[1][0].plot(x,y,label="BH",color="black",linewidth=3.0)
ax[1][0].set_yscale("log")
x,y = kde(np.loadtxt("TPSS/eigenvec1.txt"))
ax[1][1].plot(x,y,label="TPSS",color="black",linewidth=3.0)
ax[1][1].set_yscale("log")
x,y = kde(np.loadtxt("BP86/eigenvec1.txt"))
ax[1][2].plot(x,y,label="BP86",color="black",linewidth=3.0)
ax[1][2].set_yscale("log")
for i in range(2):
    for j in range(3):
        ax[i][j].set_yscale("log")
        ax[i][j].legend(frameon=False,fontsize=14)
        ax[i][j].set_xlabel(r"$x$",fontsize=16)
        ax[i][j].set_ylabel(r"$\hat{f}(x)$",fontsize=16)
        ax[i][j].tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()
plt.savefig("kde.png",dpi=300)