import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
k_B = 8.617333 * 10**(-5) # ev/K
pi = math.pi
hbar = 6.582119569 * 1e-16 # eV*s

x1 = [4e7,5e7,6e7,7e7,8e7,9e7]
for i in range(len(x1)):
    x1[i] = np.sqrt(x1[i]/100)
    
mu_PBE0 = np.loadtxt("PBE0/mu.txt")
mu_PBE = np.loadtxt("PBE/mu.txt")
mu_B3LYP = np.loadtxt("B3LYP/mu.txt")
mu_BHANDHLYP = np.loadtxt("BHANDHLYP/mu.txt")
mu_TPSS = np.loadtxt("TPSS/mu.txt")
mu_BP86 = np.loadtxt("BP86/mu.txt")

fig,ax = plt.subplots()


#ax.plot(x1,mu_k10_cor,'k^-')#,label='k=1.0')
#ax.plot(x1,mu_k0,'b--o',label='$k=0.0\,(\sigma=0.00\mathrm{eV})$',markersize=10, markerfacecolor='none')
ax.plot(x1,mu_PBE0,'r--o',label='$PBE0$',markersize=10,markerfacecolor='none')
ax.plot(x1,mu_PBE,'g--o', label='$PBE$',markersize=10,markerfacecolor='none')
ax.plot(x1,mu_B3LYP,'k--o',label='$B3LYP$',markersize=10,markerfacecolor='none')
ax.plot(x1,mu_BHANDHLYP,'--o',label='$BHANDHLYP$',markersize=10,markerfacecolor='none')
ax.plot(x1,mu_TPSS,'--o', label='$TPSS$',markersize=10,markerfacecolor='none')
ax.plot(x1,mu_BP86,'--o',label='$BP86$',markersize=10,markerfacecolor='none')

# ax.set_xlabel(r'$F^{1/2}$ (V/cm)$^{1/2}$',fontsize=12)
ax.set_xlabel('$F^{1/2} \; [(V/cm)^{1/2}]$',fontsize=14)
ax.set_ylabel("$\mu \; [cm^2/(Vs)]$",size=14)
ax.set_xlim(600,1100)
#ax.set_ylim(1e-5,1e1)
#ax.annotate("solid line: correlated", xy=(0.1,0.05), xycoords='axes fraction', size=12)
#ax.annotate("dashed line: uncorrelated", xy=(0.1,0.1), xycoords='axes fraction', size=12)
ax.set_ylim(1e-9,1e-3)
plt.yscale('log')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.legend(fontsize=10, frameon=False)
plt.savefig('PF_plot.jpg', dpi=600)
plt.savefig('PF_plot.pdf', dpi=600)