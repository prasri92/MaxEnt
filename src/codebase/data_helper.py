#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 08:30:48 2019

@author: roshanprakash
"""

import numpy as np
from itertools import product as iProd

class DataHelper():
    def __init__(self, num_diseases=10, num_clusters=5):
        self.N = num_diseases
        self.K = num_clusters
        self.clusters_1 = self.makeClusters(overlap=False)
        self.clusters_2 = self.makeClusters(overlap=True)
        self.cluster_stats_1 = self.getClusterSummary(overlap=False)
        self.cluster_stats_2 = self.getClusterSummary(overlap=True)
        self.disease_vec = list(iProd(range(2),repeat=self.N))
        
    def makeClusters(self, overlap=False):
        assert self.N>=self.K, \
        'Reduce the number of clusters. Not possible to have {} clusters'.format(self.K)
        d_idxs = np.arange(self.N)
        size = self.N//self.K
        temp = [list(d_idxs[idx*size:idx*size+size]) for idx in range(self.K)]
        rem = self.N%self.K
        if rem>0:
            temp[-1].extend(d_idxs[self.N-rem:])
        clusters = {temp.index(d):d for d in temp}
        if overlap:
            # choose m clusters randomly
            m = np.random.randint(low=max(0, self.K//2), high=max(1, self.K))
            selections = np.random.choice(self.K, replace=False, size=m)
            overlaps = {}
            for idx_1 in selections:
                # first choose a cluster to include entries from
                idx_2 = np.random.choice(np.delete(np.arange(self.K), idx_1), \
                                         replace=False, size=1)[0]
                # next, choose a random number of diseases from this cluster
                overlaps[idx_1] = np.random.choice(clusters[idx_2], replace=False, \
                                size=np.random.randint(low=1, high=len(clusters[idx_2])))
            # modify original clusters to contain overlapping elements
            for idx in selections:
                clusters[idx].extend(overlaps[idx])
        return clusters
    
    def getClusterSummary(self, overlap=False):
        if not overlap:
            clusters = self.clusters_1
        else:
            clusters = self.clusters_2
        cluster_stats = {}
        A_k = {} # exclusive diseases  
        B_k = {} # overlapping diseases
        E_k = {} # diseases not contained in any cluster 'k'
        for k in range(self.K):
            A_k[k]=[]
            B_k[k]=[]
            E_k[k] = list(np.delete(np.arange(self.N), clusters[k]))
        for d in range(self.N):
            for k in range(self.K):
                if d in clusters[k]:
                    exclusive=True
                    for k_ in np.delete(np.arange(self.K), k):
                        if d in clusters[k_]:
                            exclusive=False
                            if not d in B_k[k]:
                                B_k[k].append(d)
                    if exclusive:
                        A_k[k].append(d)
        for k in range(self.K):
            cluster_stats[k] = {'A': A_k[k], 'E': E_k[k], 'B': B_k[k]}
        return cluster_stats  
    
    def computeProbability(self, D, p=0.5):
        diseases = list(np.argwhere(D==1))
        n = len(diseases)
        alpha = (1-np.exp(-0.5*n))*(np.exp(-0.5))
        prob = 0.0
        for k in self.clusters_2.keys():
            temp = 1/self.K
            D_and_B = 0
            D_and_A = 0
            D_and_E = 0
            for idx in diseases:
                d = list(idx)[0]
                if d in self.B[k]:
                    D_and_B+=1
                if d in self.A[k]:
                    D_and_A+=1
                if d in self.E[k]:
                    D_and_E+=1
                    
            #print(D_and_E, D_and_A)
            for i in range(D_and_A):
                temp *= p/(len(self.clusters[k])-i)
            for j in range(D_and_B):
                temp *= ((p/(len(self.clusters[k])-D_and_A-j))+((1-p)/(self.N-len(self.A[k])-j)))
            for k in range(D_and_E):
                temp *= ((1-p)/(self.N-len(self.A[k])-D_and_B-k))
            #print(temp)
            prob+=temp
        return prob*alpha
        
data = DataHelper(10, 5)
#print(d.clusters_1)
#print(d.clusters_2)
print(data.cluster_stats_2)
for disease in data.disease_vec:
    #print(data.computeProbability(d))
    data.computeProbability(d)
