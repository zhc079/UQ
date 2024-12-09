from scipy.stats import wasserstein_distance
from scipy.stats import entropy
import scipy.stats as stats

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.ticker as mticker

k_B = 8.617333 * 10**(-5) # ev/K
pi = math.pi
hbar = 6.582119569 * 1e-16 # eV*s
T = 300

def Marcus(E,J,re):
    temp = 2*pi/hbar * J**2/math.sqrt(4*pi*(re*k_B*300))*math.exp(-(E-re)**2/(4*re*k_B*300) )
    return temp

tick_font_size = 13

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
    xmin = -0.9
    xmax = -0.4
    ax_histx.hist(x,bins=nbins, range=(xmin,xmax),color = "blue")
    
    ymin = -0.9
    ymax = -0.4
    width = 0.02
    ax_histy.hist(y,bins=nbins, range=(ymin,ymax), orientation='horizontal',color = "blue")

    plt.style.use('ggplot')
    #plt.grid(False)
    ax.hist2d(x, y,bins=nbins,range=[[xmin,xmax],[ymin,ymax]],norm=mpl.colors.LogNorm())
    ax.plot(np.arange(xmin,xmax,width),np.arange(ymin,ymax,width), "--",c="red")

E_000 = np.loadtxt("votcaMADN000/Edip_qmmm.txt")
E_005 = np.loadtxt("votcaMADN005/Edip_qmmm.txt")
E_010 = np.loadtxt("votcaMADN010/Edip_qmmm.txt")
E_015 = np.loadtxt("votcaMADN015/Edip_qmmm.txt")
E_020 = np.loadtxt("votcaMADN020/Edip_qmmm.txt")
E_025 = np.loadtxt("votcaMADN025/Edip_qmmm.txt")

ToFs = [0.0416,0.426,0.00263,0.0329,0.00123,0.161,0.00566,740.8,144.6,62.5]
lam = [0.388,0.303,0.375,0.494,0.310,0.304,0.505,0.495,0.312,0.493]
dict_E = { 1:E_025, 2:E_000, 3:E_005, 4:E_010, 5:E_015, 6:E_020}

dict_Jstr = { 1:r"($\alpha_{HFX}=0.05$)",2:r"($\alpha_{HFX}=0.05$)", 3:r"($\alpha_{HFX}=0.15$)", 4:r"($\alpha_{HFX}=0.23$)", 5:r"($\alpha_{HFX}=0.24$)", 6:r"($\alpha_{HFX}=0.26$)", 7:r"($\alpha_{HFX}=0.27$)", 8: r"($\alpha_{HFX}=0.35$)", 9: r"($\alpha_{HFX}=0.45$)", 10: r"($\alpha_{HFX}=0.55$)"}

dict_Jstr = { 1:r"($\alpha_{HFX}=0.25$)",2:r"($\alpha_{HFX}=0.0$)", 3:r"($\alpha_{HFX}=0.05$)",4:r"($\alpha_{HFX}=0.10$)", 5:r"($\alpha_{HFX}=0.15$)", 6:r"($\alpha_{HFX}=0.20$)", 7:r"($\alpha_{HFX}=0.23$)", 8:r"($\alpha_{HFX}=0.24$)", 9: r"($\alpha_{HFX}=0.26$)", 10: r"($\alpha_{HFX}=0.27$)", 11: r"($\alpha_{HFX}=0.35$)", 12:r"($\alpha_{HFX}=0.45$)", 13:r"($\alpha_{HFX}=0.55$)",}

fig = plt.figure(figsize=(10,6), dpi=300)
# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
'''gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.1) '''
#gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
#                      left=0.2, right=0.8, bottom=0.2, top=0.8,
#                      wspace=0.05, hspace=0.05)

gs0 = gridspec.GridSpec(2, 3, figure=fig)
for fi in range(2):
    for fj in range(3):
        if fi==1 and fj==2:
            continue
        gs00 = gridspec.GridSpecFromSubplotSpec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                            wspace=0.01, hspace=0.01, subplot_spec=gs0[fi,fj])

        # Create the Axes.
        ax00 = fig.add_subplot(gs00[1, 0])

        ax00_histx = fig.add_subplot(gs00[0, 0], sharex=ax00)
        ax00_histy = fig.add_subplot(gs00[1, 1], sharey=ax00)

        ax00.tick_params(axis='both', which='major',color="black", labelsize=tick_font_size,labelcolor="black")
        ax00.tick_params(axis='both', which='minor',color="black", labelsize=tick_font_size,labelcolor="red")

        str1 = dict_Jstr[2+fi*3+fj]
        ax00.set_xlabel(r"$E^{polar}(\alpha_{HFX}=0.25)$[eV]",color="black",fontsize=tick_font_size)
        ax00.set_ylabel(r"$E^{polar}$"+str1+"[eV]",color="black",fontsize=tick_font_size)
        #ax00.annotate("(a)", xy=(-0.5,1.25), xycoords='axes fraction', size=12)
        #ax00.annotate("no disorder", xy=(0.4,0.05), xycoords='axes fraction', size=10)
        plt.setp(ax00.spines.values(), color="black")
        plt.setp(ax00_histx.spines.values(), color="black")
        plt.setp(ax00_histy.spines.values(), color="black")
        ax00.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax00.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax00.axis("on")
        # Draw the scatter plot and marginals.
        scatter_hist(dict_E[1], dict_E[2+fi*3+fj], ax00, ax00_histx, ax00_histy)
        if fi==0 and fj==0:
            ax00.annotate("(a)", xy=(-0.55, 1.25), xycoords='axes fraction',fontsize=15) 
        if fi==0 and fj==1:
            ax00.annotate("(b)", xy=(-0.55, 1.25), xycoords='axes fraction',fontsize=15) 
        if fi==0 and fj==2:
            ax00.annotate("(c)", xy=(-0.55, 1.25), xycoords='axes fraction',fontsize=15) 
        if fi==1 and fj==0:
            ax00.annotate("(d)", xy=(-0.55, 1.25), xycoords='axes fraction',fontsize=15)
        if fi==1 and fj==1:
            ax00.annotate("(e)", xy=(-0.55, 1.25), xycoords='axes fraction',fontsize=15) 
######################
fi=1
fj=2
HFX = [0.0,0.05,0.10,0.15,0.20,0.25]
sigma_Edipqmmm = [0.044, 0.043, 0.043, 0.047, 0.045, 0.045]
gs00 = gridspec.GridSpecFromSubplotSpec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                            wspace=0.01, hspace=0.01, subplot_spec=gs0[fi,fj])
# Create the Axes.
ax00 = fig.add_subplot(gs00[1, 0])
ax00.plot(HFX, sigma_Edipqmmm, "k--o", markersize=8)
ax00.set_facecolor('white')
ax00.xaxis.set_ticks(np.arange(0.0, 0.3, 0.1))
ax00.yaxis.set_ticks(np.arange(0.04, 0.05, 0.003))
ax00.annotate("(f)", xy=(-0.55, 1.25), xycoords='axes fraction',fontsize=tick_font_size+3) 

ax00.tick_params(axis='both', which='major',color="black", labelsize=tick_font_size,labelcolor="black")
ax00.tick_params(axis='both', which='minor',color="black", labelsize=tick_font_size,labelcolor="red")
#ax00.xaxis.set_major_formatter(labelcolor="red")

ax00.set_xlabel(r"$\alpha_{HFX}$", color="black", fontsize=tick_font_size)
ax00.set_ylabel(r"$\sigma(E^{polar})$ [eV] ", color="black",fontsize=tick_font_size)

plt.setp(ax00.spines.values(), color="black")
ax00.axis("on")
######################
plt.tight_layout()
plt.savefig("scatterEdip_qmmm.pdf",dpi=400)

for i in range(1,6):
    print("--IFNO: Wasserstein Distance: ",i, wasserstein_distance(dict_E[1], dict_E[int(i+1)]))

for i in range(1,6):
    index = np.argmax(abs(dict_E[1]-dict_E[int(i+1)]))
    print("--INFO: maximum", np.max(abs(dict_E[1]-dict_E[int(i+1)])), "at index ", index, "with energy ",dict_E[int(i+1)][index], " while for alpha=0.25, energy is: ", dict_E[1][index])

for i in range(1,7):
    print("--IFNO: polarization energy range: ",i, np.max(dict_E[int(i)]), np.min(dict_E[int(i)]))