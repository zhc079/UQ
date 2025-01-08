import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

data = np.loadtxt("Sobol_Sti_E.txt")
Sti_max = np.max(data)
print(Sti_max)
print(np.argmax(np.max(data)))
########## get energy and coordinates
N=1000
posX = np.zeros(N)
print(max(posX))
posY = np.zeros(N)
posZ = np.zeros(N)
eCation = np.zeros(N)
index = np.zeros(N)

tree = ET.parse('jobs.xml')
root = tree.getroot()

for i in range(int(len(root)/3)):
    posX[i] = float(root[i*3][6][0][1].text.split()[0])
    posY[i] = float(root[i*3][6][0][1].text.split()[1])
    posZ[i] = float(root[i*3][6][0][1].text.split()[2])
    eCation[i] = float(root[i*3+2][6][0][2].text) - float(root[i*3][6][0][2].text)
    index[i] = i

nbins = 50
setupsize = 18

fig = plt.figure(figsize=(14, 6))
# First subplot with 3D projection
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax1.view_init(elev=25, azim=70, roll=1)
scatter = ax1.scatter(posX, posY,posZ, c=(np.log10(data)), cmap='viridis', s=40, alpha=0.7)  # Adjust 'cmap' as needed
cbar = plt.colorbar(scatter, ax=ax1, label=r'log$_{10}S_{T,i}$')  # Add colorbar for reference
cbar.ax.tick_params(labelsize=setupsize)
cbar.set_label(r'log$_{10}S_{T,i}$', size=setupsize)
ax1.set_xlabel(r'$X$ [nm]',fontsize=setupsize)
ax1.set_ylabel(r'$Y$ [nm]',fontsize=setupsize)
ax1.set_zlabel(r'$Z$ [nm]',fontsize=setupsize)
ax1.tick_params(axis='both', labelsize=setupsize)
ax1.annotate("(a)", xy=(-0.05,1.03), xycoords='axes fraction', size=setupsize)

ax2 = fig.add_subplot(1,2,2)
ax2.hist(np.log10(data), bins= nbins,histtype='step', linewidth=2, facecolor='c', 
         hatch='|', edgecolor='k',fill=True)
ax2.set_xlabel(r'log$_{10}S_{T,i}$', fontsize=setupsize)
ax2.set_ylabel(r'counts', fontsize=setupsize)
ax2.tick_params(axis="both",labelsize=setupsize)
ax2.annotate("(b)", xy=(-0.15,1.03), xycoords='axes fraction', size=setupsize)

# Show the plot
plt.tight_layout()
fig.savefig("fig_Sobol_E.pdf",dpi=300)
fig.savefig("fig_Sobol_E.png",dpi=300)