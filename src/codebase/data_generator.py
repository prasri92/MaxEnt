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
    def __init__(self, alpha, s, num_diseases, num_clusters, tau):
    # def __init__(self, alpha, s, num_diseases=10, num_clusters=5, tau=[0.2]*5):
        """
        A Data Generator sub-class, that operates under the <DataHelper> superclass.
        This class contains methods to synthetically generate disease data.
   
        PARAMS
        ------
        - alpha (float) : the parameter used for choosing 'n', the number of diseases 
          to sample for any instance. Parameter for truncated exponentials distribution. 
          equal to b in scipy.stats.truncexpon module
        - s (float) : the parameter for Zipf distribution; used for choosing a cluster 
          from where the in-cluster diseases will be sampled; s is the parameter used
          to define the weights in the truncated zipfian distribution
        - num_diseases(int) : the total number of possible diseases
        - num_clusters(int) : the number of clusters used for 
          grouping the diseases
        - tau (list) : the probability of choosing a cluster while sampling clusters 
          for grouping diseases ; should sum to 1.0, and
          len(<beta>) should be equal to <num_clusters>
        
        RETURNS
        -------
        None
        """
        super().__init__(num_diseases=num_diseases, num_clusters=num_clusters, tau=tau)
        self.alpha = alpha
        self.s = s
        self.num_clusters = num_clusters
             
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
        
        # first, choose 'n', using alpha - alpha should be an truncated exponential distribution
        #VERIFY
        lower, upper, scale = 1, self.N, 1 
        X = stats.truncexpon(b=(upper-lower)/scale, scale=scale)
        # X = stats.truncexpon(b=self.alpha, loc = lower, scale=scale)
        n = X.rvs(1)
        n = int(n[0])
        
        # next, choose 'k', using beta - should be a truncated zipfian distribution
        x = np.arange(1, self.num_clusters+1)
        weights = x ** (-self.s)
        weights /= weights.sum() 
        bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(x, weights))
        k = bounded_zipf.rvs(size=1)
        k = int(k[0])-1
        
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
            
            # for patients with zero diseases 
            if n == 0:
                return list(r)

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
    #set dataset_size to be number of patients 
    dataset_size = 100
    #FIX CSV WRITE
    with open("../../dataset/synthetic_data_1.csv", "a") as csvFile: 
        for i in range(dataset_size):
            # By setting alpha = 2 * num_diseases, we have make lambda for trunc exp to be 0.5 
            gen = DataGenerator(alpha=20, s=1.5, num_diseases=10, num_clusters=5, tau=[0.2]*5)
            row = gen.generate_instance(False)
            csv.writer(csvFile).writerow(row)
    csvFile.close()