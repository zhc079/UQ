from scipy.stats import wasserstein_distance
from scipy.stats import entropy
import scipy.stats as stats

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import matplotlib.gridspec as gridspec
import matplotlib as mpl

k_B = 8.617333 * 10**(-5) # ev/K
pi = math.pi
hbar = 6.582119569 * 1e-16 # eV*s
T = 300

def Marcus(E,J,re):
    temp = 2*pi/hbar * J**2/math.sqrt(4*pi*(re*k_B*300))*math.exp(-(E-re)**2/(4*re*k_B*300) )
    return temp

tick_font_size = 10

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax.set_facecolor('white')
    ax_histx.set_facecolor('white')
    ax_histy.set_facecolor('white')

    ax_histx.tick_params(axis="x", labelbottom=False,labelsize=tick_font_size)
    ax_histx.tick_params(axis="y", labelleft=False,labelsize=tick_font_size)

    ax_histy.tick_params(axis="y", labelleft=False,labelsize = tick_font_size)
    ax_histy.tick_params(axis="x", labelbottom=False,labelsize = tick_font_size)
    # plt.style.use('_mpl-gallery-nogrid')
    # ax.hist2d(x, y,bins=50,norm=mpl.colors.LogNorm())
    # fig.colorbar(ax.hist2d(x, y,bins=50,norm=mpl.colors.LogNorm())[3],location='left')
    nbins = 30
    # now determine nice limits by hand:
    xmin = -2.0
    xmax = -0.2
    ax_histx.hist(x,bins=nbins, range=(xmin,xmax),color = "blue")
    
    ymin = -2.0
    ymax = -0.2
    width = 0.1
    ax_histy.hist(y,bins=nbins, range=(ymin,ymax), orientation='horizontal',color = "blue")

    plt.style.use('ggplot')
    #plt.grid(False)
    ax.hist2d(x, y,bins=nbins,range=[[xmin,xmax],[ymin,ymax]],norm=mpl.colors.LogNorm())
    ax.plot(np.arange(xmin,xmax,width),np.arange(ymin,ymax,width), "--",c="red")

J_data = np.loadtxt("PBE/J.txt")
seg1 = J_data[:,0]
seg2 = J_data[:,1]
J_PBE0 = np.loadtxt("PBE0/J.txt")[:,2]
J_PBE = np.loadtxt("PBE/J.txt")[:,2]
J_B3LYP = np.loadtxt("B3LYP/J.txt")[:,2]
J_BH = np.loadtxt("BHANDHLYP/J.txt")[:,2]
J_TPSS = np.loadtxt("TPSS/J.txt")[:,2]
J_BP86 = np.loadtxt("BP86/J.txt")[:,2]
J_wB97X = np.loadtxt("wB97X/J.txt")[:,2]
J_wB97XD3 = np.loadtxt("wB97X-D3/J.txt")[:,2]
J_M06L = np.loadtxt("M06L/J.txt")[:,2]
J_BHLYP = np.loadtxt("BHLYP/J.txt")[:,2]

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

Es_PBE0 = np.loadtxt("PBE0/Estat.txt")
Es_PBE = np.loadtxt("PBE/Estat.txt")
Es_B3LYP = np.loadtxt("B3LYP/Estat.txt")
Es_BH = np.loadtxt("BHANDHLYP/Estat.txt")
Es_TPSS = np.loadtxt("TPSS/Estat.txt")
Es_BP86 = np.loadtxt("BP86/Estat.txt")
Es_wB97X = np.loadtxt("wB97X/Estat.txt")
Es_wB97XD3 = np.loadtxt("wB97X-D3/Estat.txt")
Es_M06L = np.loadtxt("M06L/Estat.txt")
Es_BHLYP = np.loadtxt("BHLYP/Estat.txt")

ToFs = [1.23e-3,1.04e-3,4.28e-3,737.94,1.37e-2,7.46e-3,0.02916,0.0489,0.319,2.47]
lam = [0.388,0.303,0.375,0.494,0.310,0.304,0.505,0.495,0.312,0.493]
dict_E = { 1:E_PBE0, 2:E_PBE, 3:E_B3LYP, 4:E_BH, 5:E_TPSS, 6:E_BP86, 7:E_wB97X, 8: E_wB97XD3, 9: E_M06L, 10: E_BHLYP}
dict_Es = { 1:Es_PBE0, 2:Es_PBE, 3:Es_B3LYP, 4:Es_BH, 5:Es_TPSS, 6:Es_BP86, 7:Es_wB97X, 8: Es_wB97XD3, 9: Es_M06L, 10: Es_BHLYP}
dict_J = { 1:J_PBE0, 2:J_PBE, 3:J_B3LYP, 4:J_BH, 5:J_TPSS, 6:J_BP86, 7:J_wB97X, 8: J_wB97XD3, 9: J_M06L, 10: J_BHLYP}

dict_funt = { 1:"PBE0", 2:"PBE", 3:"B3LYP", 4:"BH", 5:"TPSS", 6:"BP86", 7:"wB97X", 8: "wB97XD3", 9: "M06L", 10: "BHLYP"}

dict_Estr = { 1:"E(PBE0)", 2:"E(PBE)", 3:"E(B3LYP)", 4:"E(BH)", 5:"E(TPSS)", 6:"E(BP86)", 7:"E(wB97X)", 8: "E(wB97XD3)", 9: "E(M06L)", 10: "E(BHLYP)"}
dict_Jstr = { 1:r"(PBE0)", 2:r"(PBE)", 3:r"(B3LYP)", 4:r"(BH)", 5:r"(TPSS)", 6:r"(BP86)", 7:r"(wB97X)", 8: r"(wB97XD3)", 9: r"(M06L)", 10: r"(BHLYP)"}

fig = plt.figure(figsize=(8,8), dpi=300)
# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
'''gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.1) '''
#gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
#                      left=0.2, right=0.8, bottom=0.2, top=0.8,
#                      wspace=0.05, hspace=0.05)

gs0 = gridspec.GridSpec(3, 3, figure=fig)
for fi in range(3):
    for fj in range(3):

        gs00 = gridspec.GridSpecFromSubplotSpec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                            wspace=0.01, hspace=0.01, subplot_spec=gs0[fi,fj])

        # Create the Axes.
        ax00 = fig.add_subplot(gs00[1, 0])

        ax00_histx = fig.add_subplot(gs00[0, 0], sharex=ax00)
        ax00_histy = fig.add_subplot(gs00[1, 1], sharey=ax00)

        ax00.tick_params(axis='both', which='major', labelsize=tick_font_size)
        ax00.tick_params(axis='both', which='minor', labelsize=tick_font_size)

        str1 = dict_Estr[2+fi*3+fj]
        ax00.set_xlabel(r"E(PBE0) [eV]",fontsize=12)
        ax00.set_ylabel(str1+" [eV]",fontsize=12)
        #ax00.annotate("(a)", xy=(-0.5,1.25), xycoords='axes fraction', size=12)
        #ax00.annotate("no disorder", xy=(0.4,0.05), xycoords='axes fraction', size=10)
        plt.setp(ax00.spines.values(), color="black")
        plt.setp(ax00_histx.spines.values(), color="black")
        plt.setp(ax00_histy.spines.values(), color="black")

        ax00.axis("on")
        # Draw the scatter plot and marginals.
        scatter_hist(dict_E[1], dict_E[2+fi*3+fj], ax00, ax00_histx, ax00_histy)

plt.tight_layout()
plt.savefig("scatterE_all.pdf",dpi=500)