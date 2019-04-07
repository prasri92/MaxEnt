#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 08:30:48 2019

@author: roshanprakash
"""
import numpy as np
import time

class DataHelper():
    
    def __init__(self, num_diseases=10, num_clusters=5):
        """
        A Data Helper class, that creates clusters of diseases and stores information;
        useful for computing the probability of generation of any disease vector.
   
        PARAMS
        ------
        - num_diseases(int, default=10) : the total number of possible diseases
        - num_clusters(int, default=5) : the number of clusters used while 
        grouping the diseases
        
        RETURNS
        -------
        None
        """
        self.N = num_diseases
        self.K = num_clusters
        self.disjoint_clusters = self.makeClusters(overlap=False)
        self.overlapping_clusters = self.makeClusters(overlap=True)
        self.disjoint_clusters_stats = self.getClustersSummaries(overlap=False)
        self.overlapping_clusters_stats = self.getClustersSummaries(overlap=True)
        
    def makeClusters(self, overlap=False):
        """
        Groups the diseases into different clusters.
        
        PARAMETERS
        ----------
        - overlap(bool, default=False) : if True, overlapping clusters will be created
        
        RETURNS
        -------
        - a dictionary containing the cluster ID as key and the contained disease numbers
        (0<=n<=N) as values.
        """
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
                # choose a uniform random number of out-of-cluster diseases to add 
                overlaps[idx_1] = np.random.choice(\
                        np.delete(np.arange(self.N), clusters[idx_1]), replace=False, \
                        size=np.random.randint(low=1, high=self.N-len(clusters[idx_1])))
            # modify original clusters to contain overlapping elements
            for idx in selections:
                clusters[idx].extend(overlaps[idx])
        return clusters
    
    def getClustersSummaries(self, overlap=False):
        """
        Gathers important cluster information relative to the entire sample space of 
        diseases.
        
        PARAMETERS
        ----------
        - overlap(bool, default=False) : if True, overlapping clusters will be created
        
        RETURNS
        -------
        - a dictionary containing information regarding the A, B, and E metrics for 
        every cluster.
        
        NOTE: For any cluster,
        - A : the diseases that are exclusive to the cluster.
        - B : the diseases that are contained in the cluster 
        and in at least one other cluster.
        - E : the diseases that are not in the cluster.
        """
        if not overlap:
            clusters = self.disjoint_clusters
        else:
            clusters = self.overlapping_clusters
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
                            B_k[k].append(d)
                            exclusive=False # 'd' is no more exclusive to the cluster 'k'
                    if exclusive:
                        A_k[k].append(d)
        for k in range(self.K):
            cluster_stats[k] = {'A': A_k[k], 'E': E_k[k], 'B': B_k[k]}
        return cluster_stats  
    
    def computeProbability(self, r, p=0.5):
        """
        Computes the probability of generating a disease vector 'r'.
        
        PARAMETERS
        ----------
        - r (list) : a binary vector of size 'N', the number of diseases
        - p (float, default=0.5) : the parameter 'p' for the Binomial Random Variable 
        
        RETURNS
        -------
        - the probability of generating the disease vector 'r', according to the synthetic
        data generation scheme.
        """
        obs = list(np.argwhere(np.array(r)==1))
        #alpha = (1-np.exp(-0.05*len(obs)))*(np.exp(-0.05))
        alpha=1/self.N
        prob = 0.0
        for k in self.overlapping_clusters.keys():
            # initializations
            temp = 1/self.K
            D_and_B = 0
            D_and_A = 0
            D_and_E = 0
            # compute required stats
            for d_idx in obs:
                d = list(d_idx)[0] # d_idx is a numpy array containing one integer (look np.argwhere)
                if d in self.overlapping_clusters_stats[k]['B']:
                    D_and_B+=1
                if d in self.overlapping_clusters_stats[k]['A']:
                    D_and_A+=1
                if d in self.overlapping_clusters_stats[k]['E']:
                    D_and_E+=1
            # finally, compute the probability using these stats
            D_k = len(self.overlapping_clusters[k])
            A_k = len(self.overlapping_clusters_stats[k]['A'])
            for i in range(D_and_A):
                temp*=p/(D_k-i)
            for j in range(D_and_B):
                temp*=(p/(D_k-D_and_A-j)+((1-p)/ \
                          (self.N-A_k-j)))
            for l in range(D_and_E):
                temp*=(1-p)/(self.N-A_k-D_and_B-l)
            prob+=temp
        return prob*alpha
   
    def computeAll(self, timer=False): # add feature to pickle probs, if needed
        """
        Computes the probabilities of generation of all possible disease vectors.
        
        PARAMETERS
        ----------
        - timer (bool, default=False) : if True, records the computational time
        
        RETURNS
        -------
        - A list containing probabilities for all possible 2^N disease vectors.
        """
        probs = []
        total=0.0
        if timer:
            tic = time.time()
        for idx in range(2**self.N):
            b = format(idx, '0{}b'.format(self.N))
            r = [int(j) for j in b]
            probs.append(self.computeProbability(r))
            total+=probs[-1]
        print('Ignoring last probability, total = {}'.format(total))
        if timer:
            toc = time.time()
            print('Computational time for {} probabilities = {} seconds'.format(2**self.N, toc-tic))
        # compute the probability of last disease manually (1-sum(other P's)) ;
        # this bit sequence will contain self.N+1 bits and hence, can't be computed like others!
        probs.append(1-sum(probs))   
        return probs

if __name__=='__main__': 
    data = DataHelper(15, 4)
    p_vals = data.computeAll(timer=True)