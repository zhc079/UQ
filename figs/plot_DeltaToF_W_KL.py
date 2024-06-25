from scipy.stats import wasserstein_distance
from scipy.stats import entropy

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

J_data = np.loadtxt("PBE/J.txt")
seg1 = J_data[:,0]
seg2 = J_data[:,1]
Js = J_data[:,2]
J_PBE0 = np.loadtxt("PBE0/J.txt")
J_PBE = np.loadtxt("PBE/J.txt")
J_B3LYP = np.loadtxt("B3LYP/J.txt")
J_BH = np.loadtxt("BHANDHLYP/J.txt")
J_TPSS = np.loadtxt("TPSS/J.txt")
J_BP86 = np.loadtxt("BP86/J.txt")
J_wB97X = np.loadtxt("wB97X/J.txt")
J_wB97XD3 = np.loadtxt("wB97X-D3/J.txt")
J_M06L = np.loadtxt("M06L/J.txt")
J_BHLYP = np.loadtxt("BHLYP/J.txt")

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
dict_J = { 1:J_PBE0, 2:J_PBE, 3:J_B3LYP, 4:J_BH, 5:J_TPSS, 6:J_BP86, 7:J_wB97X, 8: J_wB97XD3, 9: J_M06L, 10: J_BHLYP}

WDs = []
delta_ToF = []
D_KL = []
for key1 in dict_E:
    for key2 in dict_E:
        #if key1 > key2:
        temp = wasserstein_distance(dict_E[key1],dict_E[key2])
        temp2 = abs(ToFs[key1-1] - ToFs[key2-1])
        WDs.append(temp)
        delta_ToF.append(temp2)
        D_KL.append((entropy(dict_E[key1],dict_E[key2])))
'''
D_KL = []
delta_ToF2 = []
for key1 in dict_E:
    for key2 in dict_E:
        if key1 != key2:
            temp2 = abs(ToFs[key1-1] - ToFs[key2-1])
            delta_ToF2.append(temp2)
            D_KL.append((entropy(dict_E[key1],dict_E[key2])))
'''


for i in range(len(WDs)):
    print("--INFO:",i,WDs[i],delta_ToF[i])
PR = sp.stats.pearsonr(WDs,delta_ToF)
print(len(WDs))
print(len(delta_ToF))
print(PR)

size =23
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,8))
ax1.scatter(WDs,delta_ToF,s=90,facecolor="None",edgecolor="black")
# ax1.scatter(WDs, delta_ToF) #,"o",markersize=9,color="black")
ax1.set_xlabel(r"$W_1(P_f(E),P_{f'}(E))$",fontsize=size)
ax1.set_ylabel(r"$\Delta $ToF",fontsize=size)
ax1.set_yscale("log")
ax1.tick_params(axis='both',labelsize=size)
ax1.annotate("(a)", xy=(-0.23,1), xycoords='axes fraction', size=30)

#ax1.legend(fontsize=size,frameon=False)

#ax2.scatter(D_KL, delta_ToF,"o",markersize=9,color="black")
ax2.scatter(D_KL,delta_ToF,s=90,facecolor="None",edgecolor="red")
ax2.set_yscale("log")
ax2.set_xlabel(r"KL$(P_f(E),P_{f'}(E))$",fontsize=size)
ax2.set_ylabel(r"$\Delta $ToF",fontsize=size)
ax2.set_yscale("log")
ax2.tick_params(axis='both',labelsize=size)
ax2.annotate("(b)", xy=(-0.21,1), xycoords='axes fraction', size=30)
#ax2.legend(fontsize=size,frameon=False)
plt.tight_layout()
fig.savefig("DeltaToF_W_KL_E.pdf",dpi=450)

'''
fig, (ax1, ax2) = plt.subplots(1, 2)#,sharey=True)
#plt.figure(figsize=(8,4))
plt.subplots_adjust(wspace=0, hspace=0)
ax1.scatter(WDs,delta_ToF,color="red",label=r"W")
ax1.set_yscale("log")
ax1.set_xlabel(r"$W_1(P_f(E),P_{f'}(E))$",fontsize=14)
ax1.set_ylabel(r"$\Delta $ToF",fontsize=14)

ax2.scatter(D_KL,delta_ToF, label="entropy")
ax2.set_yscale("log")
ax2.set_xlabel(r"KL$(P_f(E),P_{f'}(E))$",fontsize=14)
ax2.set_ylabel(r"$\Delta $ToF",fontsize=14)

plt.tight_layout()
fig.savefig("DeltaToF_W_KL_E.pdf",dpi=450)

'''
