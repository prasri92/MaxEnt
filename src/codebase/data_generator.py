#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 18:25:00 2019

@author: roshanprakash
"""
import numpy as np
import scipy.stats as stats 
from data_helper import *
import csv

class DataGenerator(DataHelper):
    
    def __init__(self, e, z, num_diseases, num_clusters, beta, p=None, q1=None, q2=None):
        """
        A Data Generator sub-class, that operates under the <DataHelper> superclass.
        This class contains methods to synthetically generate disease data.
   
        PARAMS
        ------
        - e(float) : parameter for the truncated exponential distribution ; used while 
                     choosing the number of diseases in an instance to be generated
        - z(float) : the skew for the Zipfian distribution ; used while choosing a cluster 
                     from where in-cluster diseases will be sampled
        - num_diseases(int) : the total number of possible diseases
        - num_clusters(int) : the number of clusters used for grouping the diseases
        - beta(list) : the probabilities of choosing each of the clusters while sampling
                       clusters for grouping diseases ; should sum to 1.0, and
                       len(<beta>) should be equal to <num_clusters>
        - p(float) : the binomial's 'p' value in the disjoint clusters case
        - q1(float) : the first binomial's 'p' value in the overlapping clusters case
        - q2(float) : the second binomial's 'p' value in the overlapping clusters case
    
        RETURNS
        -------
        None
        """
        self.e = e
        self.z = z
        self.num_clusters = num_clusters
        self.p = p
        assert abs(1-sum(beta))<1e-6, 'Betas do not add to 1.0, Check and try again!'
        super().__init__(alpha=self._compute_alpha(num_diseases), num_diseases=num_diseases, \
                          num_clusters=num_clusters, tau=self._compute_tau(), beta=beta, p=p, q1=q1, q2=q2)
          
    def _compute_alpha(self, N):
        """ 
        Computes the probabilities of choosing each of 0 to N diseases according to the simulation scheme.
        
        PARAMETERS
        ----------
        - N(int) : the number of diseases
        
        RETURNS
        -------
        - A list of size N+1, containing the probabilities of choosing each of 0 to N diseases.
        """
        alpha = []
        for i in range(N+1):
            alpha.append(stats.expon.pdf(i, scale=self.e))
        return np.array(alpha)/sum(alpha)
    
    def _compute_tau(self):
        """ 
        Computes the probabilities of choosing each of the <num_clusters> clusters.
        
        PARAMETERS
        ----------
        - None
        
        RETURNS
        -------
        - A list of size <num_clusters>, containing the probabilities of choosing each of the clusters.
        """
        tau = np.arange(1, self.num_clusters+1)**(-self.z)
        tau/=tau.sum()
        return tau
    
    def generate_instance(self, overlap=False):
        """
        Generates a disease vector using one of two sampling schemes.
        
        PARAMETERS
        ----------
         - overlap(bool, default=False) : if True, overlapping clusters will be accessed from 
           the super class.
         
        RETURNS
        -------
        - a binary vector of size 'N', the total number of possible diseases, wherein each bit
          indicates presence or absence of the corresponding disease.
        """
        # initializations
        D = []
        r = np.zeros(self.N)
        r = r.astype(int)
        # first, choose 'n', from a truncated exponential distribution
        n = np.random.choice(np.arange(self.N+1), p=self.alpha)
        # next, choose 'k', from a truncated zipfian distribution
        k = np.random.choice(np.arange(self.num_clusters), p=self.tau)
        if overlap: # scheme for generation from overlapping clusters
            if  n==0:
                return list(r), n
            C_k = max(min(np.random.binomial(n=n, p=self.q1), len(self.overlapping_clusters[k])), \
                        n+len(self.overlapping_clusters[k])-self.N)
            if self.overlapping_clusters_stats[k]['A']: # this cluster contains exclusive diseases
                A_k = max(min(np.random.binomial(n=C_k, p=self.q2), len(self.overlapping_clusters_stats[k]['A'])), \
                        C_k-len(self.overlapping_clusters_stats[k]['B']))
                D.extend(np.random.choice(self.overlapping_clusters_stats[k]['A'], size=A_k, replace=False))
                D.extend(np.random.choice(self.overlapping_clusters_stats[k]['B'], size=C_k-A_k, replace=False))
            else: # this cluster does NOT contain any exclusive diseases
                A_k = 0
                D.extend(np.random.choice(self.overlapping_clusters_stats[k]['B'], size=C_k, replace=False))
            D.extend(np.random.choice(self.overlapping_clusters_stats[k]['E'], size=n-C_k, replace=False))
        else: # scheme for generation from disjoint clusters
            if n==0: 
                return list(r), n
            C_k = max(min(np.random.binomial(n=n, p=self.p), len(self.disjoint_clusters[k])), \
                        n+len(self.disjoint_clusters[k])-self.N)
            D.extend(np.random.choice(self.disjoint_clusters[k], size=C_k, replace=False))
            D.extend(np.random.choice(self.disjoint_clusters_stats[k]['E'], size=n-C_k, replace=False))
        r[np.array(D)] = 1
        return list(r), n 

# =========================================================================================================================== #   

def run(file_name, dataset_size, e, z, num_diseases, num_clusters, beta, p=None, q1=None, q2=None, overlap=True):
    """ 
    Runs the synthetic data generator to generate instances and saves the generated instances to disk.
    
    PARAMETERS
    ----------
    - file_name(str) : the filename to use for saving generated data instances 
    - dataset_size(int) : the number of data instances to generate
    - e(float) : the parameter for the truncated exponential distribution
    - z(float) : the skew for the Zipfian distribution 
    - num_diseases(int) : the total number of possible diseases
    - num_clusters(int) : the number of clusters used for grouping the diseases
    - beta(list) : the probabilities of choosing each of the clusters while sampling
                       clusters for grouping diseases ; should sum to 1.0, and
                       len(<beta>) should be equal to <num_clusters>
    - p(float) : the binomial's 'p' value in the disjoint clusters case
    - q1(float) : the first binomial's 'p' value in the overlapping clusters case
    - q2(float) : the second binomial's 'p' value in the overlapping clusters case
    
    RETURNS
    -------
    - None
    """
    n = np.zeros(num_diseases)
    with open(file_name, "w") as csvFile: 
        first_row = list(np.arange(num_diseases))
        csv.writer(csvFile).writerow(first_row)
        for i in range(dataset_size):
            gen = DataGenerator(e=e, z=z, num_diseases=num_diseases, num_clusters=num_clusters, beta=beta, p=p, q1=q1, q2=q2)
            row, no_diseases = gen.generate_instance(overlap=overlap)
            n[no_diseases-1]+=1
            csv.writer(csvFile).writerow(row)
    csvFile.close()

if __name__ == '__main__':
    #example test case 
    run("../../dataset/synthetic_data_test_1.csv", 500, 0.7, 1.5, 10, 5, [0.2,0.2,0.2,0.2,0.2], 0.6, 0.3, 0.15)