#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 18:25:00 2019

@author: roshanprakash
"""
import numpy as np
from scipy.stats import planck
from data_helper import *

class DataGenerator(DataHelper):
    
    def __init__(self, alpha, s, num_diseases=10, num_clusters=5, tau=[0.2]*5):
        """
        A Data Generator sub-class, that operates under the <DataHelper> superclass.
        This class contains methods to synthetically generate disease data.
   
        PARAMS
        ------
        - alpha (float) : the parameter used for choosing 'n', the number of diseases 
          to sample for any instance.
        - s (float) : the parameter for Zipf distribution; used for choosing a cluster 
          from where the in-cluster diseases will be sampled
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
        super().__init__(num_diseases=10, num_clusters=5, tau=[0.2]*5)
        self.alpha = alpha
        self.s = s
        self.H = 0.0 # normalizing constant for the Zipf distribution
             
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
        
        # ==================== MODIFY BELOW =======================#
        # first, choose 'n', using alpha - alpha should be an exponential distribution
        n = 4
        # n = np.random.choice(np.arange(self.N)) 
        # next, choose 'k', using beta - should be a geometric distribution 
        k = np.random.choice(np.arange(self.K))
        # =========================================================#
        
        # Sample in-cluster and out-of-cluster diseases
        if overlap: # scheme for generation from overlapping clusters
            C = min(np.random.binomial(n=n, p=0.75), len(self.overlapping_clusters[k]))
            outer_diseases = list(np.delete(np.arange(self.N), self.overlapping_clusters[k]))
            if len(outer_diseases)>n-C: # sufficient number of unique out-of-cluster diseases exist
                D.extend(np.random.choice(self.overlapping_clusters[k], size=C, replace=False))
                while len(D)<n:
                    choice = np.random.choice(np.delete(np.arange(self.N), \
                                                    self.overlapping_clusters[k]))
                    if not choice in D:
                        D.append(choice) 
            else: # insufficient number of unique out-of-cluster diseases
                D.extend(outer_diseases)
                while len(D)<n:
                    # choose remaining diseases from cluster 'k'
                    choice = np.random.choice(self.overlapping_clusters[k])
                    if not choice in D:
                        D.append(choice) 
                
        else: # scheme for generation from disjoint clusters
            C = min(np.random.binomial(n=n, p=0.75), len(self.disjoint_clusters[k]))
            outer_diseases = list(np.delete(np.arange(self.N), self.disjoint_clusters[k]))
            if len(outer_diseases)>n-C: # sufficient number of out-of-cluster diseases exist
                D.extend(np.random.choice(self.disjoint_clusters[k], size=C, replace=False))
                D.extend(np.random.choice(outer_diseases, size=n-C, replace=False))
            else:
                # adjusted C = C + (n - number of out-of-cluster diseases)
                D.extend(np.random.choice(self.disjoint_clusters[k], \
                                            size= C+n-len(outer_diseases), replace=False))
                D.extend(outer_diseases) # all the out-of-cluster diseases need to be added 
        r[np.array(D)] = 1
        return list(r)
    
if __name__=='__main__':
    gen = DataGenerator(2, 1.5)
    print(gen.generate_instance(True))
    print(gen.generate_instance(False))