import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

data = np.loadtxt("Sobol_Sti_J_all.txt")
data1 = []
for i in range(len(data)):
    if data[i] > 1e-35:
        data1.append(data[i])

########## get energy and coordinates


nbins = 50
setupsize = 18

fig,ax2 = plt.subplots()
ax2.hist(np.log10(data1), bins= nbins,histtype='step', linewidth=2, facecolor='c', 
         hatch='|', edgecolor='k',fill=True)
ax2.set_xlabel(r'log$_{10}S_{T,i}$', fontsize=setupsize)
ax2.set_ylabel(r'counts', fontsize=setupsize)
ax2.tick_params(axis="both",labelsize=setupsize)
ax2.annotate("(b)", xy=(-0.15,1.03), xycoords='axes fraction', size=setupsize)

# Show the plot
plt.tight_layout()
fig.savefig("fig_Sobol_J.pdf",dpi=300)
fig.savefig("fig_Sobol_J.png",dpi=300)