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


dict_dE = {}
dict_expDE = {}
dict_expDElam = {}
dict_w = {}

for key in dict_E:
    print(key)
    dE, expDE, expDElam, w = [],[],[],[]
    for i in range(len(seg1)):
        temp = dict_E[key][int(seg1[i])] - dict_E[key][int(seg2[i])]
        temp_J = dict_J[key][int(seg1[i]),2]
        dE.append(temp)
        dE.append(-temp)
        
        #expDE.append(np.exp((temp)/(k_B * T)))
        #expDE.append(np.exp(-temp/(k_B * T)))
        expDE.append(np.exp(temp))
        expDE.append(np.exp(-temp))

        expDElam.append(np.exp(-(temp-lam[key-1])**2/(4*k_B * T*lam[key-1] )))
        expDElam.append(np.exp(-(-temp-lam[key-1])**2/(4*k_B * T*lam[key-1] )))

        w.append(Marcus(temp,temp_J,lam[key-1]))
        w.append(Marcus(-temp,temp_J,lam[key-1]))

    dict_dE[key] = dE 
    dict_expDE[key] = expDE
    dict_expDElam[key] = expDElam
    dict_w[key] = w

print("############## info: expDE and dE ############")
print(dict_expDE[1][1:10])
print(dict_dE[1][1:10])

WDs1,WDs2,WDs3,WDs4 = [],[],[],[]
delta_lam = []
delta_ToF = []
delta_2ndEigval = []
D_KL = []
for key1 in dict_dE:
    for key2 in dict_dE:
        if key1 > key2:
            print(key1,key2)
            temp = wasserstein_distance(dict_dE[key1],dict_dE[key2])
            WDs1.append(temp)

            temp = wasserstein_distance(dict_expDE[key1],dict_expDE[key2])
            WDs2.append(temp)

            temp = wasserstein_distance(dict_expDElam[key1],dict_expDElam[key2])
            WDs3.append(temp)
            
            temp = wasserstein_distance(dict_w[key1],dict_w[key2])
            WDs4.append(temp)

            temp2 = (ToFs[key1-1] - ToFs[key2-1])
            
            delta_ToF.append(temp2)
            delta_lam.append((lam[key1-1] - lam[key2-1]))

print(WDs4[1:19])
pairs = len(WDs1)
print(pairs)
tau,pvalue = stats.kendalltau(WDs1,delta_ToF)
print(tau,pvalue)
print("spearmanr dE-TOF",stats.spearmanr(WDs1,delta_ToF))
print("spearmanr e^dE-TOF",stats.spearmanr(WDs2,delta_ToF))
print(r"spearmanr $e^{(dE-\lam)}$-TOF",stats.spearmanr(WDs3,delta_ToF))
print("spearmanr w-TOF",stats.spearmanr(WDs4,delta_ToF))
print(stats.spearmanr(delta_lam,delta_ToF))


fig, ax1 = plt.subplots(nrows=2,ncols=2) #,sharey=True)
ax1[0][0].scatter(WDs1,delta_ToF,edgecolor="red",facecolor="None", label="(a)") #label=r"$P(\Delta E) - \Delta$ ToF")
ax1[0][1].scatter(WDs2,delta_ToF,edgecolor="green",facecolor="None",label="(b)") #label=r"$P(e^{\Delta E}) - \Delta$ ToF")
ax1[1][0].scatter(WDs3,delta_ToF,edgecolor="purple",facecolor="None",label="(c)")#label=r"$P(e^{\frac{(\Delta E-\lambda)^2}{4kT \lambda}}) - \Delta$ ToF")
ax1[1][1].scatter(WDs4,delta_ToF,edgecolor="black",facecolor="None",label="(d)") #,label=r"$P(\omega) - \Delta$ ToF")
for i in range(2):
    for j in range(2):
        ax1[i][j].set_yscale("log")

for ii in range(2):
    for jj in range(2):
        ax1[ii][jj].set_ylabel(r"$\Delta$ToF")

ax1[0][0].text(-0.28, 1,"(a)",transform=ax1[0][0].transAxes,fontsize=12)
ax1[0][1].text(-0.28, 1,"(b)",transform=ax1[0][1].transAxes,fontsize=12)
ax1[1][0].text(-0.28, 1,"(c)",transform=ax1[1][0].transAxes,fontsize=12)
ax1[1][1].text(-0.28, 1,"(d)",transform=ax1[1][1].transAxes,fontsize=12)

ax1[1][1].set_xscale("log")

ax1[0][0].set_xlabel(r"$W_1(P_f (\Delta E), P_{f'} (\Delta E))$")
ax1[0][1].set_xlabel(r"$W_1(P_f ( e^{(\Delta E)}), P_{f'} (e^{(\Delta E)}))$")
ax1[1][0].set_xlabel(r"$W_1(P_f ( e^{\frac{(\Delta E - \lambda)^2}{4 K_B T \lambda}} ), P_{f'} (e^{\frac{(\Delta E - \lambda)^2}{4 K_B T \lambda}}))$")
ax1[1][1].set_xlabel(r"$W_1(P_f (\omega), P_{f'} (\omega))$")

plt.subplots_adjust(wspace=0)
plt.tight_layout()
fig.savefig("DeltaToF_W_all.pdf",dpi=300)

fig, ax1 = plt.subplots(nrows=2,ncols=2,sharey=True)
ax1[0][0].hist(dict_w[1],label="PBE0",bins=100)
ax1[0][1].hist(dict_w[2],label="PBE",bins=100)
ax1[1][0].hist(dict_w[3],label="B3LYP",bins=100)
ax1[1][1].hist(dict_w[4],label="BH",bins=100)
for i in range(2):
    for j in range(2):
        ax1[i][j].set_yscale("log")
#fig.savefig("distri_w.jpg",dpi=150)
    
######## plot empirical distrib #####################
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points
    n = len(data)
    # x-data for the ECDF
    x = np.sort(data)
    # y-data for the ECDF
    y = np.arange(1, n+1) / n
    return x, y

fig,[ax,ax2] = plt.subplots(1,2,figsize=(16,8))
x,y = ecdf(dict_w[1])
ax.plot(x,y,label="PBE0")
x,y = ecdf(dict_w[2])
ax.plot(x,y,label="PBE")
x,y = ecdf(dict_w[3])
ax.plot(x,y,label="B3LYP")
x,y = ecdf(dict_w[4])
ax.plot(x,y,label="BH")
x,y = ecdf(dict_w[5])
ax.plot(x,y,label="TPSS")
x,y = ecdf(dict_w[6])
ax.plot(x,y,label="BP86")

x,y = ecdf(dict_w[1])
ax2.plot(x,y,label="PBE0")
x,y = ecdf(dict_w[2])
ax2.plot(x,y,label="PBE")
x,y = ecdf(dict_w[3])
ax2.plot(x,y,label="B3LYP")
x,y = ecdf(dict_w[4])
ax2.plot(x,y,label="BH")
x,y = ecdf(dict_w[5])
ax2.plot(x,y,label="TPSS")
x,y = ecdf(dict_w[6])
ax2.plot(x,y,label="BP86")
sizes=25
ax.set_xscale("log")
ax.set_xlabel(r"$\omega_{i,j} [s^{-1}]$",fontsize=sizes)
ax.set_ylabel(r"CDF($\omega$)",fontsize=sizes)
ax.legend()
ax2.set_yscale("log")
ax2.set_xscale("log")
ax2.set_xlabel(r"$\omega_{i,j} [s^{-1}]$",fontsize=sizes)
ax2.set_ylabel(r"CDF($\omega$)",fontsize=sizes)
ax.legend(fontsize=19,frameon=False)
ax.tick_params(axis='both', labelsize=sizes)
ax2.tick_params(axis='both', labelsize=sizes)
plt.tight_layout()
fig.savefig("cdf_w_log.pdf",dpi=500)

'''
########## plot some rate distributions ########################################
fig,ax = plt.subplots()
nbin = 100
ax.hist(dict_w[1],label="PBE0",histtype='step',bins=100)
#ax.hist(dict_w[2],label="PBE",histtype='step',bins=100)
#ax.hist(dict_w[3],label="B3LYP",histtype='step',bins=100)
ax.hist(dict_w[2],label="PBE",histtype='step',bins=100)
ax.set_yscale("log")
#ax.set_xscale("log")
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"counts")
ax.legend()
fig.savefig("distri2_w.jpg",dpi=150)
'''
