#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
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

 
 

k_B = 8.617333 * 10**(-5) # ev/K
pi = math.pi
hbar = 6.582119569 * 1e-16 # eV*s


def report(xk):
    frame = inspect.currentframe().f_back
    if frame.f_locals['iter_']%100 == 0:
        print("    ---INFO SOLVER, iteration:", frame.f_locals['iter_'], "residual:", frame.f_locals['resid'])

#rate Matrix calculator
def calculate_1c_ratematrix(J2, E1, lam, n_sites, seg1, seg2, drX, drY, drZ, efield):

    W1 = np.zeros((int(n_sites),int(n_sites)))

    fieldVector = [efield,0,0] # field in x direction
            
    for i in range(len(seg1)):
        if seg1[i] < seg2[i]:
            # marcus rate seg2[i] -> seg1[i]
            dE_field = 1E-9*(fieldVector[0]*drX[i] + fieldVector[1]*drY[i] + fieldVector[2]*drZ[i])
            dE = dE_field  +  (E1[seg2[i]]-E1[seg1[i]])
            W1[seg2[i],seg1[i]] = Marcus(dE, J2[i], lam)
            W1[seg1[i],seg2[i]] = Marcus(-dE, J2[i], lam)

    return W1

# marcus rate
def Marcus(dE,Jsq,re):
    temp = 2*pi/hbar * Jsq/math.sqrt(4*pi*(re*k_B*T))*math.exp(-(dE-re)**2/(4*re*k_B*T) )
    return temp


def parse_SQL(statefile):

    con = sqlite3.connect(statefile)
    c=con.cursor()
    id = []
    posX = []
    posY = []
    posZ = []
    eCation = []
    sql = "SELECT id, posX, posY, posZ, eCation FROM segments"

    c.execute(sql)
    for row in c.fetchall():
        id.append(row[0])
        posX.append(float(row[1]))
        posY.append(float(row[2]))
        posZ.append(float(row[3]))
        eCation.append(float(row[4]))
    
    # get pairs (only i->j, for rates, add j->i)
    seg1 = []
    seg2 = []
    drX = []
    drY = []
    drZ = []
    J2 = []
    sql ="SELECT seg1, seg2, drX, drY, drZ, Jeff2h FROM pairs"
    c.execute(sql)

    for row in c.fetchall():
        seg1.append(row[0])
        seg2.append(row[1])
        drX.append(float(row[2]))
        drY.append(float(row[3]))
        drZ.append(float(row[4]))
        J2.append(float(row[5]))

    sql = "SELECT box11, box22, box33 FROM frames"
    c.execute(sql)
    for row in c.fetchall():
        boxX = float(row[0])
        boxY = float(row[1])
        boxZ = float(row[2])
    
    return id, posX, posY, posZ, eCation, seg1, seg2, J2, drX, drY, drZ, boxX, boxY, boxZ

def construct_regular_3Dlattice(L, usePBC, latconst=1.0):
    size = L[0]*L[1]*L[2]
    latY = L[1]*latconst
    latZ = L[2]*latconst
    posX = np.zeros((L[0],L[1],L[2]))
    posY = np.zeros((L[0],L[1],L[2]))
    posZ = np.zeros((L[0],L[1],L[2]))
    for i in range(L[0]):
        for j in range(L[1]):
            for k in range(L[2]):
                posX[i,j,k] = i + 0.5
                posY[i,j,k] = j + 0.5
                posZ[i,j,k] = k + 0.5

    posX = latconst*posX
    posY = latconst*posY
    posZ = latconst*posZ
            
    seg1 = []
    seg2 = []

    posX = posX.reshape(size)
    posY = posY.reshape(size)
    posZ = posZ.reshape(size)

    drX = []
    drY = []
    drZ = []

    for i in range(size):
        for j in range(size):
            dX = posX[j]-posX[i]
            if usePBC:
                for boxY in range(-1,2):
                    dY = posY[j] + boxY*latY -posY[i]
                    for boxZ in range(-1,2):
                        dZ = posZ[j]+ boxZ*latZ -posZ[i]
                        if np.sqrt(dX**2 + dY**2 + dZ**2) <1.5:
                            seg1.append(i)
                            seg2.append(j)
                            drX.append(dX) # 1nm lattice constant
                            drY.append(dY)
                            drZ.append(dZ)  
            else:
                dY = posY[j] -posY[i]
                dZ = posZ[j] -posZ[i]
                if np.sqrt(dX**2 + dY**2 + dZ**2) <1.5:
                    seg1.append(i)
                    seg2.append(j)
                    drX.append(dX) # 1nm lattice constant
                    drY.append(dY)
                    drZ.append(dZ)

    
    return size, posX, posY, posZ, seg1, seg2, drX, drY, drZ


# function to calculate the spatial energy correlation function
def spatial_correlation_function(energies, dist_mat, resolution=0.1):

    # calculate AVG
    AVG = np.mean(energies)
    # calcuate VAR
    VAR = np.var(energies)
    # calculate STD
    STD = np.std(energies)


    distances = []
    C_values = []
    NDATA = len(energies)
    # for each segment
    for i in range(NDATA):
        row_distance = dist_mat.getrow(i)
        for j in row_distance.nonzero()[1]:
            distances.append(dist_mat[i,j])
            C_values.append( (energies[i]-AVG) * (energies[j]-AVG))


    # make histogram of Cs ( not really a histrogram, just add C to bins)
    #print("Correlations over ", len(C_values), "pairs")
    MIN = np.min(distances)
    MAX = np.max(distances)
    #print("minimum distance in pairs: ", MIN)
    #print("maximum distance in pairs: ", MAX)
    # total number of bins
    BIN = int(( MAX - MIN ) / resolution +0.5) +1 
    #print("Total number of bins", BIN )

    histCs = [] 
    for i in range(BIN):
        histCs.append([])

    #print(len(histCs))
    for i in range(len(C_values)):
        this_bin = int((distances[i]-MIN) / resolution +0.5)
        #print(distances[i], C_values[i]) 
        histCs[this_bin].append(C_values[i])
        # average of all Cs in bin, divided by VAR  

    R = []
    C = []
    D = []
    for i in range(BIN):
        corr = 0.0
        dcorr2 = 0.0
        for j in range(len(histCs[i])):
            #print(i,j, histCs[i][j])
            corr = corr + histCs[i][j]
        corr = corr / VAR / len(histCs[i])
        #for j in range(len(histCs[i])):
        #    dcorr2 = dcorr2 + (histCs[i][j] / VAR / float(len(histCs[i])) - corr)**2
        #dcorr2 = dcorr2 / float(len(histCs[i])) / float(len(histCs[i]) -1)
        R.append(MIN + float(i) * resolution)
        C.append(corr)
        D.append(np.sqrt(dcorr2))

    return R,C,D



# function to label the 2 carrier cases
def f_index(N, x,y):
    '''
    N: the last index 
    x,y: integer between [0,N]
    '''
    if x<y:
        temp = int((N + N-x+1)*x/2 + y-x)-1
    elif y<x:
        temp = int((N + N-y+1)*y/2 + x-y)-1
    else:
        print("got x=y, should this have happened? N = ", N, "x = ", x, "y= ", y )
    return temp

def extended_outflowrate_vector_batched(G, n1, n2, total_batches, batch_id):
    items_per_batch = int(G.number_of_edges()/total_batches)
    this_batch_start = batch_id * items_per_batch
    if batch_id == total_batches -1: 
        this_batch_stop = G.number_of_edges()-1
    else:
        this_batch_stop = (batch_id+1) * items_per_batch

    #ar_ext = scipy.sparse.coo_matrix((n2, n2))
    idx = []
    idy = []
    outflow = []
    for u,v,a in list(G.edges(data=True))[this_batch_start:this_batch_stop]:
        #print(u,v,a)
        if u < v:
            for u_prime in range(v):
                if (u_prime != u):
                    index1 = f_index(n1-1,u_prime,v)
                    index2 = f_index(n1-1,u_prime,u)
                    idx.extend([index2,index1])
                    idy.extend([index1,index2])
                    outflow.extend([a['weight'],G.edges[v, u]['weight']])
            for v_prime in np.arange(v+1,n1,1):
                index1 = f_index(n1-1,v,v_prime)
                index2 = f_index(n1-1,u,v_prime)
                idx.extend([index2,index1])
                idy.extend([index1,index2])
                outflow.extend([a['weight'],G.edges[v, u]['weight']])
    
    return scipy.sparse.coo_array((outflow, (idx, idy)), shape=(n2, n2))

def extended_outflowrate_vector(G, n1, n2):
    ar_ext = scipy.sparse.coo_matrix((n2, n2))

    pool_obj = multiprocessing.Pool(total_batches)
    ar_ext_batched = pool_obj.map(partial(extended_outflowrate_vector_batched, G, n1, n2, total_batches),range(total_batches))

    for ar_ext_batch in ar_ext_batched:
        ar_ext = ar_ext + ar_ext_batch

    pool_obj.close()


    outflowRate = np.array(np.sum(ar_ext,axis=0)).reshape(n2) # it is a matrix
    #print('outflowRate of extended matrix completed!')
    return outflowRate


def extended_probability_matrix_batched(outflowrate, G, n1, n2, sink_region, total_batches, batch_id):

    items_per_batch = int(G.number_of_edges()/total_batches)
    this_batch_start = batch_id * items_per_batch
    if batch_id == total_batches -1: 
        this_batch_stop = G.number_of_edges()-1
    else:
        this_batch_stop = (batch_id+1) * items_per_batch

    ##print(multiprocessing.current_process(), "start ", this_batch_start, " stop", this_batch_stop)
    idx = []
    idy = []
    A = []
    for u,v,a in list(G.edges(data=True))[this_batch_start:this_batch_stop]:
        if u in sink_region:
            continue
        if v in sink_region:
            continue
        rate = a['weight']
        rate_back = G.edges[v, u]['weight']
        if u < v:
            for u_prime in range(v):
                if u_prime in sink_region:
                    continue
                if (u_prime != u):
                    index1 = f_index(n1-1,u_prime,v)
                    index2 = f_index(n1-1,u_prime,u)
                    idx.extend([index2,index1])
                    idy.extend([index1,index2])
                    A.extend([ rate / outflowrate[index1], rate_back / outflowrate[index2] ])
            for v_prime in np.arange(v+1,n1,1):
                if v_prime in sink_region:
                    continue
                index1 = f_index(n1-1,v,v_prime)
                index2 = f_index(n1-1,u,v_prime)
                idx.extend([index2,index1])
                idy.extend([index1,index2])
                A.extend([rate / outflowrate[index1], rate_back / outflowrate[index2] ])

    return scipy.sparse.coo_array((A, (idx, idy)), shape=(n2, n2))

def extended_probability_matrix(outflowrate, G, n1, n2, sink_region):
    A = scipy.sparse.identity(n2,format='coo')
    print("    ---INFO: Total number of 1c edges:", G.number_of_edges() )

    pool_obj = multiprocessing.Pool(total_batches)
    A_batched = pool_obj.map(partial(extended_probability_matrix_batched, outflowrate, G, n1, n2, sink_region, total_batches),range(total_batches))

    for A_batch in A_batched:
        A = A - A_batch

    pool_obj.close()

    return A.T

def setup_2c_system(W1, sink_region, Sink):
    # determine the two carrier state matrix in sparse from with the help of graph methods in networkx
    n1 = W1.shape[0]
    n2 = math.comb(W1.shape[0],2)
    G = nx.from_numpy_array(W1, create_using=nx.DiGraph())

    outflowrate = extended_outflowrate_vector(G, n1, n2)
    print("    ---INFO", datetime.now().strftime("%H:%M:%S"), "outflow rate vector" )

    probability = extended_probability_matrix(outflowrate, G, n1, n2, sink_region)
    print("    ---INFO", datetime.now().strftime("%H:%M:%S"), "probability matrix" )

    # do this componentwise instead of loop
    Time_init = np.reciprocal(outflowrate)
    for i in Sink:
        Time_init[i]=0.0
    #for i in fake_site:
    #   Time_init[i]=0.0
    print("    ---INFO", datetime.now().strftime("%H:%M:%S"), "t0 vector" )

    probability = probability.tocsr()

    return probability, Time_init, G

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


def postprocess_VOTCA_to_TOF( posX, posY, posZ, seg1, seg2, J2, drX, drY, drZ, boxX, boxY, boxZ):

    # VOTCA data has PBCs and we need to define Source and Sink regions
    # some sanity checks first: are all segments inside the box
    for i in range(len(posX)):
        if posX[i] > boxX or posY[i] > boxY or posZ[i] > boxZ:
            print("segment ", i, "outside of box") 

    # due to PBC there may be connections across boundaries in x-direction, we need to identify/remove them
    pbc_pairs = []
    seg1_cleaned = []
    seg2_cleaned = []
    J2_cleaned = []
    drX_cleaned = []
    drY_cleaned = []
    drZ_cleaned = []

    for i in range(len(seg1)):
        # check distance according to posX difference
        distanceX = posX[seg2[i]-1] - posX[seg1[i]-1]
        if distanceX > drX[i]:
            #print("Found PBC pair ", seg1[i], ":", seg2[i] )
            pbc_pairs.append(i)
        else: 
            seg1_cleaned.append(seg1[i]-1)
            seg2_cleaned.append(seg2[i]-1)
            J2_cleaned.append(J2[i])
            drX_cleaned.append(drX[i])
            drY_cleaned.append(drY[i])
            drZ_cleaned.append(drZ[i])

    return pbc_pairs, seg1_cleaned, seg2_cleaned, J2_cleaned, drX_cleaned, drY_cleaned, drZ_cleaned

## construct an initial guess
def initial_guess(G, n_states, sink_region, Sink):
    rate_connToSink = []
    n1 = G.number_of_nodes()
    for u,v,a in G.edges(data=True):
        #print(u,v,a)
        if u < v and (v in sink_region):
            rate_connToSink.append(a['weight'])
    rate_max = max(rate_connToSink)
    t0 = np.array([1./rate_max]*n_states)
    for i in Sink:
        t0[i] = 0
    print("finished constructing an minimum initial guess")

    return t0

def distance_matrix_PBC(X,Y,Z,boxX,boxY,boxZ):

    NDATA = len(X)
    distmat = scipy.sparse.lil_matrix((NDATA,NDATA))

    # for each segment
    for i in range(NDATA):
        # for all other segments with larger index
        for j in range(i+1,NDATA):
            #print("Working on pair: ", i,j)
            min_dist = 1e7
            for kx in range(-1,2):
                dx = X[i] - X[j] - kx*boxX
                for ky in range(-1,2):
                    dy = Y[i] - Y[j] - ky*boxY
                    for kz in range(-1,2):
                        dz = Z[i] - Z[j] - kz*boxZ
                        dist = np.sqrt(dx**2 + dy**2 + dz**2)
                        if dist < min_dist:
                            min_dist = dist
                        if dist < 0.01:
                            print("Found weridly short distance:", dist, "between sites:", i, j)
                            print(dx, dy, dz)
                            print(X[i], X[j])
                            print(Y[i], Y[j])
                            print(Z[i], Z[j])
                            exit(1)
            # only makes sense if distance < L/2
            if (min_dist < max(boxX,boxY,boxZ)/2.):
                distmat[i,j] = min_dist
                distmat[j,i] = min_dist

    return distmat


def build_spatial_correlations(Ma,Mb,Mc,dist):
    wa = 0.2
    wb = 0.4
    lb = 9
    lc = 200 # from PRB

    N = len(Ma)
    E = np.zeros(N)
    for i in range(N):
        E1 = np.sqrt(wa)*Ma[i]
        #print(i, Ma[i])
        # get energies sorted according to distances from i
        dlist = []
        idx_list = []
        for j in dist_mat.getrow(i).nonzero()[1]:
            dlist.append(dist_mat[i,j])
            idx_list.append(j)
        ind = np.argsort(np.array(dlist))
        idx_list = np.array(idx_list)
        #sort idx_list according to idx
        idx_sorted = idx_list[ind]
        #print(idx_sorted)
        Mb_list = Mb[idx_sorted]
        #print(Mb_list)
        Mc_list = Mc[idx_sorted]

        # E2
        E2 = Mb[i]
        for j in range(lb):
            E2 = E2 + Mb_list[j]
        E2 = E2 * np.sqrt(wb/float(lb))
        
        # E3
        E3 = Mc[i]
        for j in range(lc):
            E3 = E3 + Mc_list[j]
        E3 = E3 * np.sqrt((1-wa-wb)/float(lc))
        E[i] = E1 + E2 + E3
    
    return E


DESCRIPTION = "Random Walk TOF for 2 charge carriers"

"""Read the command line arguments."""
parser = argparse.ArgumentParser()

# Add the arguments to the parser
parser.add_argument("-m", "--mode", default='VOTCA', help="Which graph type to use.", choices=["VOTCA","lattice"])
parser.add_argument("-T", "--temp", default=300, help="Temperature in K")
parser.add_argument("-p", "--parallel", default="1", help="number of parallel threads")
parser.add_argument("-r", "--reorg", default=0.23, help="Reorganization energy in eV")
parser.add_argument("-l", "--lattice", default=[8, 8, 8], help="tuple for lattice specification", action='store', nargs=3,type=int)
parser.add_argument("-e", "--efield", default=3e7, help="electric field in x-direction in V/m")
parser.add_argument("-g", "--genE", default=False, action="store_true", help="Generate new Gaussian distributed site energies for lattice")
parser.add_argument("--pbc", default=False, action="store_true", help="Using PBCs in y and z direction in lattice model.")
parser.add_argument("-n", "--numc", default=2, help="Number of carriers")
parser.add_argument("--corr", default=False, action="store_true", help="Site energies with spatial correlation")
args = parser.parse_args()


total_batches = int(args.parallel) # testing
mode=str(args.mode)
T=float(args.temp)
lam = float(args.reorg)
lat_dim = args.lattice
efield = float(args.efield)
genEnergies = args.genE
usePBC = args.pbc
N_carriers = int(args.numc)
correlated = args.corr

# some simulation parameters
print("+SIM Temperature:", T, "K")
print("+SIM Reorganization energy:", lam, "eV")
print("+SIM Electric field in x-direction:", efield, "V/m")
print("+SIM Number of carriers:", N_carriers)
print("+SIM Number of parallel threads:", total_batches)
if mode == 'lattice':
    print("+SIM Generation of Gaussian energies requested:", genEnergies)
    print("+SIM                 with spatial correlations:", correlated)
    print("+SIM Using PBC with lattice:", usePBC)

# parse the original VOTCA data
id, posX, posY, posZ, eCation, seg1, seg2, J2, drX, drY, drZ, boxX, boxY, boxZ = parse_SQL("dft_state_snap1.sql")

#R,C,D = spatial_correlation_function(eCation,posX,posY,posZ,boxX,boxY,boxZ)
#plt.plot(R,C)
#plt.savefig("corr.png")

#exit()


if mode == 'VOTCA':
    print("+SIM using plain VOTCA data")

    # post-process the VOTCA data -> check for PBC and remove PBCs in x-direction
    pbc_pairs, seg1_TOF, seg2_TOF, J2_TOF, drX_TOF, drY_TOF, drZ_TOF = postprocess_VOTCA_to_TOF( posX, posY, posZ, seg1, seg2, J2, drX, drY, drZ, boxX, boxY, boxZ)
    n_sites = len(posX)
    E_TOF = np.array(eCation)
    print("    ---INFO Removed ", len(pbc_pairs), " PBC pairs")


elif mode == 'lattice':
    print("+SIM using lattice model with topology", lat_dim[0], 'x', lat_dim[1],'x',lat_dim[2])

    # statistic of original VOTCA data
    Eave = np.average(eCation)
    Estd = np.std(eCation)
    J2ave = np.average(J2)
    J2std = np.std(J2)

    # create regular lattice and NBL
    n_sites, posX, posY, posZ, seg1_TOF, seg2_TOF, drX_TOF, drY_TOF, drZ_TOF = construct_regular_3Dlattice(lat_dim, usePBC)
    print("    ---INFO Regular lattice with lattice constant 1nm setup." )

    dist_mat = distance_matrix_PBC(posX,posY,posZ,lat_dim[0], lat_dim[1],lat_dim[2])
    #assign site energies from normal distribution or read from file
    
    if genEnergies == False and os.path.exists("E1.txt") == False:
        genEnergies = True
        print("    ---INFO reading of site enegies requested, but file not found. Requesting generation instead." )
    if genEnergies:
        if correlated:
            M1 = np.random.normal(loc=0.0, scale=Estd, size = n_sites)  
            M2 = np.random.normal(loc=0.0, scale=Estd, size = n_sites)  
            M3 = np.random.normal(loc=0.0, scale=Estd, size = n_sites)  
            E_TOF = build_spatial_correlations(M1,M2,M3,dist_mat)
        else:
            E_TOF = np.random.normal(loc=Eave, scale=Estd, size = n_sites)  
        print("    ---INFO generated", n_sites, "site energies with mean", Eave, " sigma", Estd)
        np.savetxt("E1.txt",E_TOF)
    else:
        E_TOF = np.loadtxt("E1.txt")
        print("    ---INFO READ", n_sites, "site energies with mean", Eave, " sigma", Estd)
        if len(E_TOF) != n_sites:
            print("    ---INFO Number of stored energies in E1.txt not equal to number of points in lattice. Reassigning from normal distribution!")
            if correlated:
                M1 = np.random.normal(loc=0.0, scale=Estd, size = n_sites)  
                M2 = np.random.normal(loc=0.0, scale=Estd, size = n_sites)  
                M3 = np.random.normal(loc=0.0, scale=Estd, size = n_sites)  
                E_TOF = build_spatial_correlations(M1,M2,M3,dist_mat)
            else:
                E_TOF = np.random.normal(loc=Eave, scale=Estd, size = n_sites)              
            print("    ---INFO generated", n_sites, "site energies with mean", Eave, " sigma", Estd)
            np.savetxt("E1.txt",E_TOF)

    R,C,D = spatial_correlation_function(E_TOF,dist_mat)
    plt.plot(R,C)
    plt.savefig("corr_lattice.png")
    
    # assigning a constant J2 and a exponentiall scaled version for next neaerest NBs
    J2_TOF = np.zeros(len(seg1_TOF))
    for i in range(len(seg2_TOF)):
        J2_TOF[i] = J2ave*np.exp(-(1.0-np.sqrt(drX_TOF[i]**2+drY_TOF[i]**2+drZ_TOF[i]**2)))

    # boxX length equivalent
    boxX = float(lat_dim[0])

n_states = math.comb(n_sites,N_carriers)
print("    ---INFO", "Number of sites:", n_sites )
print("    ---INFO", "Number of states:",n_states  )


# make lists of segments in Source and Sink regions
sink_width = 0.5
source_width = 0.5
source_region = []
sink_region = []
source_cut = source_width # all segments with x-position smaller than this are considered in source region
sink_cut = boxX - sink_width # all segments with x-positions larger than this are considered sink region
for i in range(len(posX)):
    if posX[i] <= source_cut:
        source_region.append(i)
    elif posX[i] >= sink_cut:
        sink_region.append(i)

print("    ---INFO Number of sites in source region [ 0 -", source_cut, "]nm:", len(source_region) )
print("    ---INFO Number of sites in sink region [", sink_cut, "- ", boxX,"]nm:", len(sink_region) )


# prepare the source states
Source = []

if N_carriers ==2:
    for subset in itertools.combinations(source_region, N_carriers):
        list_temp = list(subset)
        index_temp = f_index(n_sites-1,list_temp[0],list_temp[1])
        Source.append(index_temp)
elif N_carriers ==1:
    Source = source_region
else:
    print("+SIM ERROR! number of carriers not supported!", N_carriers)
    exit(1)

n_source = len(Source)
print("    ---INFO Number of STATES in source region [ 0 -", source_cut, "]nm:", n_source )
# would be nice to avoid this
Sink = []
if N_carriers == 2:
    for sink_site in sink_region:
        for i in np.arange(0,n_sites-1,1):
            if i != sink_site:
                Sink.append(f_index(n_sites-1,i,sink_site))
elif N_carriers == 1:
    Sink = sink_region
n_sink = len(Sink)
print("    ---INFO Number of STATES in sink region [ 0 -", sink_cut, "]nm:", n_sink )

# use some initial guess?
x0=None

for step in range(11):
    scale=0.1*step
    #print("Running energies with scaling", scale )
    W1 = calculate_1c_ratematrix(J2_TOF, scale*E_TOF, lam, n_sites, seg1_TOF, seg2_TOF, drX_TOF, drY_TOF, drZ_TOF, efield) 
    print("    ---INFO", datetime.now().strftime("%H:%M:%S"), "calculated single-carrier rate matrix" )

    # setup extended space matrices
    if N_carriers == 2:
        A, b, G = setup_2c_system(W1, sink_region, Sink)
    elif N_carriers == 1:
        A, b, G = setup_1c_system(W1,sink_region)
    else:
        print("+SIM ERROR! number of carriers not supported!", N_carriers)
        exit(1)

    # try initial guess
    #x0 = initial_guess(G,n_states, sink_region, Sink)
    # try preconditioner
    #prec = spilu(A.tocsc())
    #M = scipy.sparse.linalg.LinearOperator((n_states,n_states), prec.solve)
    maxiter = 10000
    tol=1e-5
    thres = tol*scipy.linalg.norm(b)
    print("    ---INFO SOLVER convergence thres ", thres)
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
    T_abs = []
    T_abs_serial = 0.0
    T_abs_parallel = 0.0
    for i in Source:
        this_T = Time_cont[i]
        T_abs.append(this_T)
        T_abs_serial = T_abs_serial + this_T
        T_abs_parallel = T_abs_parallel + 1./this_T

    T_abs_serial = T_abs_serial/float(n_source)
    T_abs_parallel = float(n_source)/T_abs_parallel


    # print("+SIM scale", scale, "exit code ", exit_code, " absorption time from first source state: ", T_abs[0], " from source region: ",T_abs_parallel  )
    print("+SIM scale", scale, "exit code ", exit_code, " from source region: ",T_abs_parallel  )
    print("=========================================================================================")
    print("")
    #exit(0)
    #np.savetxt("time_2RW_ver2.txt", Time_cont)




