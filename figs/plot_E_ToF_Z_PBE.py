from sklearn.cluster import KMeans
from sklearn.neighbors import RadiusNeighborsClassifier
import numpy as np
import math
import matplotlib.pyplot as plt

E66 = np.loadtxt("PBE/E66_range.txt")
all_cost = np.loadtxt("PBE/all_cost.txt")
all_ToF = np.loadtxt("PBE/all_ToF.txt")

size =23
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,8))
ln1 = ax1.plot(E66, all_cost,"--o",markersize=9,color="red", label=r"$E_{t}$ vs $Z$")
ax1.set_xlabel(r"$E_{t}$[eV]",fontsize=size)
ax1.set_ylabel(r"$Z_{2c}(E_t)$",fontsize=size)
ax1.set_yscale("log")
ax1.tick_params(axis='both',labelsize=size)
ax1.set_ylim(1e-18,1e3)
ax1.annotate("(a)", xy=(-0.25,1), xycoords='axes fraction', size=30)

#ax1.legend(fontsize=size,frameon=False)

ax1_2 = ax1.twinx()
ln1_2 = ax1_2.plot(E66,all_ToF,"--o",markersize=9,color="black",label=r"$E_{t}$ vs $ToF$")
ax1_2.set_yscale("log")
ax1_2.set_ylabel(r"$ToF(E_{t})$[s]",fontsize=size,color="red")
ax1_2.tick_params(axis='y',labelsize=size,color="red")
#ax1_2.spines['right'].set_color('red') 
lns = ln1+ln1_2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=1,fontsize=21,frameon=False)

ax2.plot(all_ToF, all_cost,"--*",markersize=9,color="black", label=r"$E_{t}$ vs $Z$")
ax2.set_xlabel(r"$ToF(E_t)$ [s]",fontsize=size)
ax2.set_ylabel(r"$Z_{2c}(E_t)$",fontsize=size)
ax2.set_yscale("log")
ax2.set_xscale("log")
ax2.tick_params(axis='both',labelsize=size)
ax2.annotate("(b)", xy=(-0.25,1), xycoords='axes fraction', size=30)
#ax2.legend(fontsize=size,frameon=False)
plt.tight_layout()
fig.savefig("fig_E_ToF_Z_nonsym.pdf",dpi=400)


############## symmetric L_w
E66 = np.loadtxt("PBE/E66_range_sym.txt")
all_cost = np.loadtxt("PBE/all_cost_sym.txt")
all_ToF = np.loadtxt("PBE/all_ToF_sym.txt")


size =23
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,8))
ln1 = ax1.plot(E66, all_cost,"--o",markersize=9,color="red", label=r"$E_{t}$ vs $Z$")
ax1.set_xlabel(r"$E_{t}$[eV]",fontsize=size)
ax1.set_ylabel(r"$Z_{2c}(E_t)$",fontsize=size)
ax1.set_yscale("log")
ax1.tick_params(axis='both',labelsize=size)
ax1.set_ylim(1e-18,1e3)
ax1.annotate("(a)", xy=(-0.25,1), xycoords='axes fraction', size=30)

#ax1.legend(fontsize=size,frameon=False)

ax1_2 = ax1.twinx()
ln1_2 = ax1_2.plot(E66,all_ToF,"--o",markersize=9,color="black",label=r"$E_{t}$ vs $ToF$")
ax1_2.set_yscale("log")
ax1_2.set_ylabel(r"$ToF(E_{t})$[s]",fontsize=size,color="red")
ax1_2.tick_params(axis='y',labelsize=size,color="red")
#ax1_2.spines['right'].set_color('red') 
lns = ln1+ln1_2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=1,fontsize=21,frameon=False)

ax2.plot(all_ToF, all_cost,"--*",markersize=9,color="black", label=r"$E_{t}$ vs $Z$")
ax2.set_xlabel(r"$ToF(E_t)$ [s]",fontsize=size)
ax2.set_ylabel(r"$Z_{2c}(E_t)$",fontsize=size)
ax2.set_yscale("log")
ax2.set_xscale("log")
ax2.tick_params(axis='both',labelsize=size)
ax2.annotate("(b)", xy=(-0.25,1), xycoords='axes fraction', size=30)
plt.tight_layout()
fig.savefig("fig_E_ToF_Z_sym.pdf",dpi=400)