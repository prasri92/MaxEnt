#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 08:30:48 2019

@author: roshanprakash
"""
from scipy.stats import binom 
from scipy.special import comb
import numpy as np
import pickle
np.random.seed(0)
import time

class DataHelper:
    
    def __init__(self, num_diseases=10, num_clusters=5, tau=[0.2]*5, beta=[0.2]*5):
        """
        A Data Helper class, that creates clusters of diseases and stores information;
        useful for computing the probability of generation of any disease vector.
   
        PARAMS
        ------
        - num_diseases(int, default=10) : the total number of possible diseases
        - num_clusters(int, default=5) : the number of clusters used for 
          grouping the diseases
        - tau (list, default=[0.2, 0.2, 0.2, 0.2, 0.2]) : the probability of choosing
          a cluster while sampling clusters for grouping diseases ; should sum to 1.0, and
          len(<beta>) should be equal to <num_clusters>
        
        RETURNS
        -------
        None
        """
        assert len(tau)==num_clusters, \
        'Incorrect number of beta parameters! Make sure beta is available for every cluster.'
        assert abs(sum(tau)-1.0)<=1e-10, \
        'Invalid beta parameters! Should be normalized between 0 and 1, and sum to 1.0!'
        self.N = num_diseases
        self.K = num_clusters
        self.tau = tau
        self.beta = beta
        self.disjoint_clusters = self.makeClusters(overlap=False)
        self.overlapping_clusters = self.makeClusters(overlap=True)
        self.disjoint_clusters_stats = self.getClustersSummaries(overlap=False)
        self.overlapping_clusters_stats = self.getClustersSummaries(overlap=True)
        
    
    def makeClusters(self, overlap=False):
        """
        Groups the diseases into different clusters.

        PARAMETERS
        ----------
        - overlap(bool, default=False) : if True, overlapping clusters 
           will be created

        RETURNS
        -------
        - a dictionary containing the cluster ID as key and the 
           contained disease numbers
          (0<=n<=N) as values.
        """
        assert self.N>=self.K, \
        'Reduce the number of clusters. Not possible to have {} clusters'.format(self.K)
        d_idxs = np.arange(self.N)
        redo = True
        #redos=0
        while redo==True:
            clusters = {idx:[] for idx in range(self.K)}
            for d_idx in d_idxs:
                if overlap:
                    if self.K==2:
                        low=2
                    else:
                        low=1
                    # choose 'm', the number of clusters this disease can belong to, randomly
                    m = np.random.randint(low=low, high=self.K+1)
                else:
                    # choose only one cluster, since every cluster should be disjoint
                    m = 1
                # choose 'm' clusters, without replacement, according to beta vector
                selections = np.random.choice(np.arange(self.K), size=m, p=self.beta, replace=False)
                for k in selections:
                    clusters[k].append(d_idx)
            for k in clusters.keys():
                if len(list(clusters[k]))!=0:
                    redo = False
                else:
                    redo = True
                    #redos+=1
                    break
        #print(clusters)
        return clusters
        '''
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
        """
        TODO: Fix the generation scheme for the clusters
        """
        
        # assert self.N>=self.K, \
        #  'Reduce the number of clusters. Not possible to have {} clusters'.format(self.K)
        # d_idxs = np.arange(self.N)
        # clusters = {idx:[] for idx in range(self.K)}
        # for d_idx in d_idxs:
        #     if overlap:
        #         # choose 'm', the number of clusters this disease can belong to, randomly
        #         m = np.random.randint(low=1, high=self.K+1)
        #     else:
        #         # choose only one cluster, since every cluster should be disjoint
        #         m = 1
        #     # choose 'm' clusters, without replacement, according to beta vector
        #     selections = np.random.choice(np.arange(self.K), size=m, p=self.beta, replace=False)
        #     for k in selections:
        #         clusters[k].append(d_idx)
        # # TODO: this has to repeat until we get no empty cluster,
        # # computational efficiency vs. change in the cluster size 
        # # usually occurs for disjoint case
        # # print('clusters are:', clusters)
        # return clusters
        
        #Example, non-overlapping clusters with equal sizes 
        return {0:[0], 1:[1], 2:[2,3], 3:[4]}
        # return {0:[0,1,2,6,4,5],1:[0,3,4,5],2:[2,5,6,7,8],3:[9,1,7,10,11]}
        '''
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
                        if d in clusters[k_] and exclusive==True:
                            B_k[k].append(d)
                            exclusive=False # 'd' is no more exclusive to the cluster 'k'
                    if exclusive:
                        A_k[k].append(d)
        for k in range(self.K):
            cluster_stats[k] = {'A': A_k[k], 'E': E_k[k], 'B': B_k[k]}
        # print(cluster_stats)
        return cluster_stats  

        
    def computeProbability(self, r, overlap=False):
        """
        Computes the probability of generating a disease vector 'r'.
        
        PARAMETERS
        ----------
        - r (list) : a binary vector of size 'N', the number of diseases
        
        RETURNS
        -------
        - the probability of generating the disease vector 'r', according to the synthetic
          data generation scheme.
        """
        obs = list(np.argwhere(np.array(r)==1))
        alpha=1/(self.N+1)
        prob = 0.0
        if obs:
            if overlap:
                for k in self.overlapping_clusters.keys():
                    # initializations
                    temp = self.tau[k]
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
                        temp*=(0.2/(D_k-i))
                    for j in range(D_and_B):
                        temp*=((0.2/(D_k-D_and_A-j))+((1-0.2)/(self.N-A_k-j)))
                    for l in range(D_and_E):
                        temp*=((1-0.2)/((self.N-A_k)-D_and_B-l))  
                    prob+=temp
            else:
                for k in self.disjoint_clusters.keys():
                    temp = self.tau[k]
                    j = 0
                    for d_idx in obs:
                        d = list(d_idx)[0]
                        if d in self.disjoint_clusters[k]:
                            j+=1 # 'j' is the number of diseases that are in D and D_k
                    size = len(self.disjoint_clusters[k])
                    if j<size:
                        b = binom.pmf(j, len(obs), p=0.75)
                    elif j==size:
                        b = 1-binom.cdf(j-1, len(obs), p=0.75)
                    a = [list(d_idx)[0] for d_idx in obs]
                    for i in self.disjoint_clusters[k]:
                        if i in a:
                            a.remove(i)
                    c = np.delete(np.arange(self.N), self.disjoint_clusters[k]).size
                    temp*=(b*(1/comb(size, j))*(1/comb(c, len(a))))
                    prob+=temp    
        else:
            prob=1
        return prob*alpha
   
    def computeAll(self, overlap=False, timer=False): # add feature to pickle probs, if needed
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
            p = self.computeProbability(r,overlap=False)
            probs.append(p)
            # probs.append(self.computeProbability(r, overlap=True))
            total+=probs[-1]
        print('Sum of probabilities = {}'.format(total))
        if timer:
            toc = time.time()
            print('Computational time for {} probabilities = {} seconds'.format(2**self.N, toc-tic))
        return probs

if __name__=='__main__': 
    data = DataHelper(20, 4, tau=[0.45, 0.25, 0.2, 0.1], beta=[0.35, 0.35, 0.15, 0.15])
    p_vals = data.computeAll(timer=True)
    outfilename = '../../output/top20diseases_synthetic.pickle'
    with open(outfilename, "wb") as outfile:
        pickle.dump(p_vals, outfile)

        