import math
k_B = 8.617333 * 10**(-5) # ev/K
pi = math.pi
hbar = 6.582119569 * 1e-16 # eV*s
###### import packages ####################################
import os
import xml.etree.ElementTree as ET

import numpy as np
import argparse
import glob, os
import sqlite3
import inspect

import math

import scipy as sp
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import *

import scipy.sparse
import networkx as nx

from functools import partial
import itertools
from itertools import repeat
import  multiprocessing 

from datetime import datetime

import os.path
import os.path
from sklearn.cluster import KMeans
from sklearn.neighbors import RadiusNeighborsClassifier

############ need to change re
re = 0.155563575 + 0.147270720
N = 1000

def report(xk):
    frame = inspect.currentframe().f_back
    if frame.f_locals['iter_']%100 == 0:
        print("    ---INFO SOLVER, iteration:", frame.f_locals['iter_'], "residual:", frame.f_locals['resid'])

def Marcus(E,J,re):
    temp = 2*pi/hbar * J**2/math.sqrt(4*pi*(re*k_B*300))*math.exp(-(E-re)**2/(4*re*k_B*300) )
    return temp

def setup_1c_system(W1, sink_region):
    # determine the two carrier state matrix in sparse from with the help of graph methods in networkx
    n1 = W1.shape[0]
    G = nx.from_numpy_array(W1, create_using=nx.DiGraph())

    outflowrate = W1.sum(axis=0)
    print("    ---INFO", datetime.now().strftime("%H:%M:%S"), "outflow rate vector" )

    normalized_rate = W1 @ np.diag(np.reciprocal(outflowrate))
    # set sink parts to zero
    for sink in sink_region:
        normalized_rate[:,sink] = 0.0

    probability = np.identity(n1) - normalized_rate.T
    print("    ---INFO", datetime.now().strftime("%H:%M:%S"), "probability matrix" )

    # do this componentwise instead of loop
    Time_init = np.reciprocal(outflowrate)
    for sink in sink_region:
        Time_init[i]=0.0
    #for i in fake_site:
    #   Time_init[i]=0.0
    print("    ---INFO", datetime.now().strftime("%H:%M:%S"), "t0 vector" )

    return scipy.sparse.csr_matrix(probability), Time_init, G


########## get energy and coordinates
posX = np.zeros(N)
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


####### need to change file name
J = np.loadtxt("J.txt")

E66_temp = eCation[66]
all_ToF = []
all_cost = []
de_range=np.arange(-1.8,2.8,0.1)
for gap in de_range:
    eCation[66] = E66_temp + gap
    W1 = np.zeros((N,N))
    for i in range(J.shape[0]):
        index1 = int(J[i,0])
        index2 = int(J[i,1])
        de = eCation[index1] - eCation[index2]
        #de = 0
        W1[index2,index1] = Marcus(de,J[i,2],re)
        W1[index1,index2] = Marcus(-de,J[i,2],re)
    # print(index1, W1[index1,index2])

    #### find the sink region
    print(np.min(posX),np.max(posX))
    cut1 = 0.5
    Source = []
    Sink = []
    for i in range(N):
        #print(posX[i])
        if posX[i] < np.min(posX)+0.5:
            Source.append(i)
        if posX[i] > np.max(posX)-0.5:
            Sink.append(i)

############## calculate eigen values and vectors ################
    n_site = W1.shape[0]
    W2 = W1+W1.transpose()
    #W2  = W1
    W_colSum = np.sum(W2,axis=0)
    D_inv = np.zeros((n_site,n_site))
    for i in range(W1.shape[0]):
        D_inv[i,i] = 1/W_colSum[i]
        #W1[i,i] = -W_colSum[i]
    L_rw = np.eye(N) - D_inv @ W2
    eigval,eigvec = np.linalg.eig(L_rw)
    #eigval,eigvec = np.linalg.eig(np.diag(W_colSum)-W2)
    eigval_sort = np.sort(eigval)
    print(eigval_sort[0:7])
    index_i = 0

    for i in range(N):
        if abs(eigval[i] - eigval_sort[1]) < 0.0005*eigval_sort[1]:
            print("found index:",i,eigval[i])
            index_i = i
    eigvec1 = eigvec[:,index_i]
    eigvect_sort = np.sort(eigvec1)
    #if np.min(eigvect_sort) > -0.5:
    #    print("--info: large entries in eigenvec: ", eigvect_sort[-1],eigvect_sort[-2])        
    #else:
    #    print("--info: small entries in eigenvec: ", eigvect_sort[0],eigvect_sort[1])
    
    data = np.zeros((N,2))
    data[:,0] = eigvec1[:]
    data[:,1] = 1
    kmeans = KMeans(n_clusters=2).fit(data)  # ,n_init="auto"
    #print(kmeans.cluster_centers_[0])
    #print(kmeans.cluster_centers_[0][0])
    #print(kmeans.labels_)
    cluster0 = np.where(kmeans.labels_ == 0)[0]
    cluster1 = np.where(kmeans.labels_ == 1)[0] # sth like array[66, 738]
    #cluster2 = np.where(kmeans.labels_ == 2)[0]
    
    center0 = kmeans.cluster_centers_[0][0]
    center1 = kmeans.cluster_centers_[1][0]
    #center2 = kmeans.cluster_centers_[2][0]
    print("----INFO cluster1: ",cluster1,center0,center1,data[cluster1[0],0])
    cost = 0
    for ii in range(cluster0.shape[0]):
        cost += (data[cluster0[ii],0] - center0 )**2
    for ii in range(cluster1.shape[0]):
        cost += (data[cluster1[ii],0] - center1 )**2
    #for ii in range(cluster2.shape[0]):
    #    cost += (data[cluster2[ii],0] - center2 )**2
    all_cost.append(cost)
    print("----INFO cost: ",cost)


    ### delete X periodic boundary
    smallX = []
    largeX = []
    for i in range(N):
        #print(posX[i])
        if (posX[i] < 2.5):
            smallX.append(i)
        if posX[i] > 6.5:
            largeX.append(i)
    print(len(smallX),len(largeX))

    for ind1 in smallX:
        for ind2 in largeX:
            W1[ind1,ind2] = 0
            W1[ind2,ind1] = 0

    
    A, b, G = setup_1c_system(W1,Sink)

    maxiter = 10000
    tol=1e-8
    thres = tol*scipy.linalg.norm(b)
    print("    ---INFO SOLVER convergence thres ", thres)
    x0=None
    Time_cont, exit_code = qmr(A, b,x0=x0, tol=tol, atol=thres, maxiter=maxiter, callback=report)
    if exit_code < 0:
        print("    ---INFO QMR SOLVER did not converge within maximum number of iteration!")
        print("    ---INFO trying BICGSTAB SOLVER with ILU preconditioning... will take some time!")
        prec = spilu(A.tocsc())
        M = scipy.sparse.linalg.LinearOperator((n_states,n_states), prec.solve)
        Time_cont, exit_code = bicgstab(A, b,x0=Time_cont, tol=tol, M=M, atol=thres, maxiter=maxiter, callback=report)
    print("    ---INFO", datetime.now().strftime("%H:%M:%S"), "LIN SYS solved" )
    x0=Time_cont


    # Average absorption time over all source states
    n_source = len(Source)
    T_abs = []
    T_abs_serial = 0.0
    T_abs_parallel = 0.0
    for i in Source:
        this_T = Time_cont[i]
        T_abs.append(this_T)
        #T_abs_serial = T_abs_serial + this_T
        T_abs_parallel = T_abs_parallel + 1./this_T

    #T_abs_serial = T_abs_serial/float(n_source)
    T_abs_parallel = float(n_source)/T_abs_parallel
    print(T_abs_parallel)
    all_ToF.append(T_abs_parallel)
    
    
print(all_cost)
print(all_ToF)
#np.savetxt("all_ToF.txt",all_ToF)

size =20 
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,8))
ax1.plot(E66_temp +de_range, all_cost,"-^",label=r"$E_{66}$ vs $Z$")
ax1.set_xlabel(r"$E_{66} [eV]$",fontsize=size)
ax1.set_ylabel(r"$Z$",fontsize=size)
ax1.set_yscale("log")
ax1.tick_params(axis='both',labelsize=size)
ax1.legend(fontsize=size)
ax2.plot(all_ToF, all_cost,"-o",label=r"$ToF$ vs $Z$")
ax2.set_xlabel(r"ToF [s]",fontsize=size)
ax2.set_ylabel(r"$Z$",fontsize=size)
ax2.set_yscale("log")
ax2.set_xscale("log")
ax2.tick_params(axis='both',labelsize=size)
ax2.legend(fontsize=size)
plt.tight_layout()
fig.savefig("fig_E_ToF_Z2.png",dpi=400)


