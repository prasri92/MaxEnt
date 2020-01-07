#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 08:30:48 2019

@author: roshanprakash
"""
from scipy.stats import binom 
from scipy.special import comb
import scipy.stats as stats
import numpy as np
import pickle
# np.random.seed(0)
import time

import random 
import bisect 
import math 
from functools import reduce 


class DataHelper:
    
    def __init__(self, num_diseases, num_clusters, alpha, tau, beta, p=None, q1=None, q2=None):
        """
        A Data Helper class, that creates clusters of diseases and stores information;
        useful for computing the probability of generation of any disease vector.
   
        PARAMS
        ------
        - num_diseases(int) : the total number of possible diseases
        - num_clusters(int) : the number of clusters used for grouping the diseases
        - alpha(list of length <num_diseases>+1) : the probability of choosing 'k' diseases 
                                                 in the synthetic generator for all values of k 
                                                 where 0<=k<=N
        - tau(list) : the probabilities of choosing each of 
                         the <num_clusters> clusters
                         while generating a disease vector ; 
                         should sum to 1.0, and
                         len(<tau>) should be equal to <num_clusters>
        - beta(list) : the probabilities of choosing each of 
                          the <num_clusters> clusters while sampling
                          clusters for grouping diseases ; 
                          should sum to 1.0, and len(<beta>) should be 
                          equal to <num_clusters>
        - p(float) : the binomial's 'p' value in the disjoint case
        - q1(float) : the first binomial's 'p' value in the overlapping case
        - q2(float) : the second binomial's 'p' value in the overlapping case
        
        RETURNS
        -------
        None
        """
        assert len(tau)==num_clusters, \
        'Incorrect number of tau parameters! Make sure beta is available for every cluster.'
        assert abs(sum(tau)-1.0)<=1e-10, \
        'Invalid tau parameters! Should be normalized between 0 and 1, and sum to 1.0!'
        self.N = num_diseases
        self.K = num_clusters
        self.alpha = alpha
        self.tau = tau
        self.beta = beta
        self.q1 = q1
        self.q2 = q2
        self.p = p
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
        redo = True
        while redo==True:
            clusters = {idx:[] for idx in range(self.K)}
            for d_idx in d_idxs:
                if overlap:
                    if self.K==2:
                        low=2
                    else:
                        low=1
                    # choose 'm', the number of clusters this disease can belong to uniformly
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
                    break
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
                        if d in clusters[k_] and exclusive==True:
                            B_k[k].append(d)
                            exclusive=False # 'd' is no more exclusive to the cluster 'k'
                    if exclusive:
                        A_k[k].append(d)
        for k in range(self.K):
            cluster_stats[k] = {'A': A_k[k], 'E': E_k[k], 'B': B_k[k]}
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
        observation = list(np.argwhere(np.array(r)==1))
        if observation:
            prob = 0.0
            if overlap:
                for k in self.overlapping_clusters.keys():
                    p_k = self.tau[k]
                    D = len(observation)
                    size = len(self.overlapping_clusters[k])
                    Ak = len(self.overlapping_clusters_stats[k]['A'])
                    D_Ak = 0
                    D_Bk = 0
                    D_Dk = 0
                    D_Dk_ = 0
                    for d_idx in observation:
                        d = list(d_idx)[0] # d_idx is a numpy array containing one integer (look np.argwhere)
                        if d in self.overlapping_clusters[k]:
                            D_Dk+=1
                        else:
                            D_Dk_+=1
                        if d in self.overlapping_clusters_stats[k]['B']:
                            D_Bk+=1
                        elif d in self.overlapping_clusters_stats[k]['A']:
                            D_Ak+=1
                    Bk = np.setdiff1d(self.overlapping_clusters[k], self.overlapping_clusters_stats[k]['A']).size
                    Ek = np.setdiff1d(np.arange(self.N), self.overlapping_clusters[k]).size 
                    d = min(size, D) 
                    if D_Dk<d:
                        b = binom.pmf(D_Dk, D, self.q1)
                        if D_Dk_==Ek: # accounts for revisits 
                            for i in range(0, D_Dk):
                                if self.N-size<D-i:
                                    b+=binom.pmf(i, D, self.q1)  
                    else:
                        b = 1-binom.cdf(D_Dk-1, D, self.q1)
                        if D_Dk_==Ek: # accounts for revisits 
                            for i in range(0, d+1):
                                if self.N-size<D-i:
                                    b+=binom.pmf(i, D, self.q1)
                    # NOTE :  Revisits (above) happen only when all out_of_cluster diseases are present in D
                    d_ = min(D_Dk, Ak)
                    if D_Ak<d_:
                        b_exc = binom.pmf(D_Ak, D_Dk, self.q2) 
                        if Bk==D_Bk: # accounts for revisits 
                            for j in range(0, D_Ak):
                                if Bk<D_Dk-j:
                                    b_exc+=binom.pmf(j, D_Dk, self.q2)
                    else:
                        b_exc = 1-binom.cdf(D_Ak-1, D_Dk, self.q2)
                        if Bk==D_Bk: # accounts for revisits 
                            for j in range(0, d_+1):
                                if Bk<D_Dk-j:
                                    b_exc+=binom.pmf(j, D_Dk, self.q2)
                    # NOTE :  Revisits (above) happen only when all overlapping diseases are present in D
                    p_k*=(b*b_exc*(1/comb(Ak, D_Ak))*(1/comb(Bk, D_Bk))\
                               *(1/comb(Ek, D_Dk_)))
                    prob+=p_k
            else:
                for k in self.disjoint_clusters.keys():
                    temp = self.tau[k]
                    j = 0
                    for d_idx in observation:
                        d = list(d_idx)[0]
                        if d in self.disjoint_clusters[k]:
                            j+=1 # 'j' is the number of diseases that are in D and D_k
                    size = len(self.disjoint_clusters[k])
                    if j<size:
                        b = binom.pmf(j, len(observation), p=self.p)
                    elif j==size:
                        b = 1-binom.cdf(j-1, len(observation), p=self.p)
                    a = [list(d_idx)[0] for d_idx in observation]
                    for i in self.disjoint_clusters[k]:
                        if i in a:
                            a.remove(i)
                    c = np.delete(np.arange(self.N), self.disjoint_clusters[k]).size
                    temp*=(b*(1/comb(size, j))*(1/comb(c, len(a))))
                    prob+=temp    
        else:
            prob=1.0
        return prob*self.alpha[len(observation)]
   
    def computeAll(self, overlap=False, timer=False): 
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
        marginal_probs = np.zeros(self.N+1)
        total=0.0
        if timer:
            tic = time.time()
        for idx in range(2**self.N):
            b = format(idx, '0{}b'.format(self.N))
            r = [int(j) for j in b]
            p = self.computeProbability(r, overlap=overlap)
            m = sum(r)
            marginal_probs[m]+=p
            probs.append(p)
            total+=probs[-1]
        print('Sum of probabilities = {}'.format(total))
        if timer:
            toc = time.time()
            print('Computational time for {} probabilities = {} seconds'.format(2**self.N, toc-tic))
        return probs,marginal_probs
'''
#non-overlapping case 
def run(outfilename, d, c, e, tau, beta, p):
    alpha = []
    for i in range(d+1):
        alpha.append(stats.expon.pdf(i, scale=e))
    alpha = np.array(alpha)/sum(alpha)
    data = DataHelper(d, c, alpha, tau, beta, p)
    p_vals = data.computeAll(timer=True, overlap=False)
    with open(outfilename, "wb") as outfile:
        pickle.dump(p_vals, outfile)
'''
#overlapping case
def run(outfilename, d, c, e, tau, beta, p, q1, q2):
    alpha = []
    for i in range(d+1):
        alpha.append(stats.expon.pdf(i, scale=e))
    alpha = np.array(alpha)/sum(alpha)
    data = DataHelper(d, c, alpha, tau, beta, p, q1, q2)
    p_vals, marginals = data.computeAll(timer=True, overlap=True)
    with open(outfilename, "wb") as outfile:
        pickle.dump((p_vals, marginals), outfile)

if __name__=='__main__': 
    # example case
    outfilename = '../../output/test_data_helper_overlap.pickle'
    run(outfilename, 10, 4, 2.4, [0.1, 0.5, 0.1, 0.3], [0.25,0.25,0.25,0.25], 0.5, 0.6, 0.7)
    # overlap
    # data = DataHelper(10, 3, alpha=[1/11]*11, tau=[0.7, 0.25, 0.05], beta=[1/3]*3, q1=0.65, q2=0.97)
    # disjoint
    #data = DataHelper(5, 4, alpha=[0.4, 0.1, 0.2, 0.1, 0.1, 0.1], tau=[0.25]*4, beta=[0.25]*4, p=0.7)
    # p_vals = data.computeAll(timer=True)
    