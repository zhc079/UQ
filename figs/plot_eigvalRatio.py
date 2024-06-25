from scipy.stats import wasserstein_distance
from scipy.stats import entropy
import scipy.stats as stats

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
k_B = 8.617333 * 10**(-5) # ev/K
pi = math.pi
hbar = 6.582119569 * 1e-16 # eV*s
T = 300

def Marcus(E,J,re):
    temp = 2*pi/hbar * J**2/math.sqrt(4*pi*(re*k_B*300))*math.exp(-(E-re)**2/(4*re*k_B*300) )
    return temp

J_data = np.loadtxt("PBE/J.txt")
seg1 = J_data[:,0]
seg2 = J_data[:,1]
Js = J_data[:,2]
E_PBE0 = np.loadtxt("PBE0/E.txt")
E_PBE = np.loadtxt("PBE/E_PBE.txt")
E_B3LYP = np.loadtxt("B3LYP/E_B3LYP.txt")
E_BH = np.loadtxt("BHANDHLYP/E.txt")
E_TPSS = np.loadtxt("TPSS/E.txt")
E_BP86 = np.loadtxt("BP86/E_BP86.txt")
E_wB97X = np.loadtxt("wB97X/E_wB97X.txt")
E_wB97XD3 = np.loadtxt("wB97X-D3/E.txt")
E_M06L = np.loadtxt("M06L/E.txt")
E_BHLYP = np.loadtxt("BHLYP/E.txt")
ToFs = [1.23e-3,1.04e-3,4.28e-3,737.94,1.37e-2,7.46e-3,0.02916,0.0489,0.319,2.47]
lam = [0.388,0.303,0.375,0.494,0.310,0.304,0.505,0.495,0.312,0.493]
dict_E = { 1:E_PBE0, 2:E_PBE, 3:E_B3LYP, 4:E_BH, 5:E_TPSS, 6:E_BP86, 7:E_wB97X, 8: E_wB97XD3, 9: E_M06L, 10: E_BHLYP}

#eigval_2nd = [3.08e10,6.47e10,1.25e11,6.65e09,6.16e+10,7.35e10,1.74e+10]
eigv_2nd_rw = [4.45e-6,5.12e-7,4.23e-6,9.90e-9,1.64e-5,5.38e-6,4.63e-7,1.52e-6,3.31e-6,4.55e-6]
eigval_2nd_W = [5.91e+04,4.198e+05,1.834e+04,2.67e+0,2.18e+05,2.07e+04, 3.32e+02, 1.65e3,4.76e3,7.17e3 ]



dict_dE = {}
dict_expDE = {}
dict_expDElam = {}
dict_w = {}

for key in dict_E:
    dE, expDE, expDElam, w = [],[],[],[]
    for i in range(len(seg1)):
        temp = dict_E[key][int(seg1[i])] - dict_E[key][int(seg2[i])]
        dE.append(temp)
        dE.append(-temp)
        
        expDE.append(np.exp(temp))
        expDE.append(np.exp(-temp))

        expDElam.append(np.exp(-(temp-lam[key-1])**2/(4*k_B * T*lam[key-1] )))
        expDElam.append(np.exp(-(-temp-lam[key-1])**2/(4*k_B * T*lam[key-1] )))

        w.append(Marcus(temp,Js[int(seg1[i])],lam[key-1]))
        w.append(Marcus(-temp,Js[int(seg1[i])],lam[key-1]))

    dict_dE[key] = dE 
    dict_expDE[key] = expDE
    dict_expDElam[key] = expDElam
    dict_w[key] = w

WDs1,WDs2,WDs3,WDs4 = [],[],[],[]
delta_lam = []
delta_ToF = []
delta_2ndEigval = []
ratio_2ndEigval_rw = []
D_KL = []
ratio_2ndEigval_W = []
for key1 in dict_E:
    for key2 in dict_E:
            if int(key1) < int(key2):
                temp2 = (ToFs[key1-1] / ToFs[key2-1])
            #temp2 = (ToFs[key1-1] / ToFs[key2-1])
            
                delta_ToF.append(temp2)
                delta_lam.append((lam[key1-1] - lam[key2-1]))
                #delta_2ndEigval.append(eigval_2nd[key1-1]-eigval_2nd[key2-1])
                #delta_2ndEigval.append(eigval_2nd[key1-1]/eigval_2nd[key2-1])
                ratio_2ndEigval_rw.append(eigv_2nd_rw[key1-1]/eigv_2nd_rw[key2-1])
                ratio_2ndEigval_W.append(eigval_2nd_W[key1-1]/eigval_2nd_W[key2-1])

print(delta_2ndEigval)
pairs = len(WDs1)
print(pairs)
#print(stats.spearmanr(delta_2ndEigval,delta_ToF))
print("-----INFO: spearmanr of 2nd eigenval Lw and ToF",stats.spearmanr(ratio_2ndEigval_W,delta_ToF))
print("-----INFO: spearmanr of 2nd eigenval Lrw and ToF",stats.spearmanr(ratio_2ndEigval_rw,delta_ToF))


fig, ax1 = plt.subplots()
plt.subplots_adjust(wspace=0, hspace=0)
#ax1.scatter(delta_2ndEigval,delta_ToF,color="red",label=r"W")
ax1.scatter(ratio_2ndEigval_rw,delta_ToF,marker="o",s=40, facecolors='none', edgecolors='r',label=r"$L_W$")
ax1.scatter(ratio_2ndEigval_W,delta_ToF,marker="*",s=40,color="blue",label = r"$L_{RW}$")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel(r"$\lambda'_{2}(f) / \lambda'_{2}(f')$",fontsize=16)
ax1.set_ylabel(r"ToF$_f$/ToF$_{f'}$",fontsize=16)
ax1.legend()
#plt.tight_layout()
fig.savefig("ratio_tof_2ndEigval.pdf",dpi=300)
#fig.savefig("ratio_tof_2ndEigval_RW.pdf",dpi=300)
