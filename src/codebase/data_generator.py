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
import matplotlib.pyplot as plt 

class DataGenerator(DataHelper):
    def __init__(self, alpha, s, num_diseases, num_clusters, tau, beta, p):
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
        - p : the binomial probability 
        - s : the skew for the Zipfian distribution

        RETURNS
        -------
        None
        """
        super().__init__(num_diseases=num_diseases, num_clusters=num_clusters, tau=tau, beta=beta, p=p)
        self.alpha = alpha
        self.s = s
        self.num_clusters = num_clusters
        self.p = p
             
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
        lower, upper, scale = 0, self.N, 1 
        def trunc_expon_rv(lower, upper, scale, size):
            cdf = np.random.uniform(stats.expon.cdf(x=lower, scale=scale),\
            stats.expon.cdf(x=upper, scale=scale), size=size)
            return stats.expon.ppf(q=cdf, scale=scale)
        n = trunc_expon_rv(lower, upper, self.alpha, 1)
        n = int(n[0])
        
        # next, choose 'k', from a truncated zipfian distribution
        x = np.arange(1, self.num_clusters+1)
        weights = x ** (-self.s)
        weights /= weights.sum() 
        bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(x, weights))
        k = bounded_zipf.rvs(size=1)
        k = int(k[0])-1
        
        # Sample in-cluster and out-of-cluster diseases
        if overlap: # scheme for generation from overlapping clusters
            C = min(np.random.binomial(n=n, p=self.p), len(self.overlapping_clusters[k]))
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
            if n == 0: # for patients with zero diseases 
                return list(r), n
            # print("cluster size", len(self.disjoint_clusters[k]))
            C = min(np.random.binomial(n=n, p=self.p), len(self.disjoint_clusters[k]))
            # print("in cluster wanted", C)
            outer_diseases = list(np.delete(np.arange(self.N), self.disjoint_clusters[k]))
            if len(outer_diseases)>n-C: # sufficient number of out-of-cluster diseases exist
                D.extend(np.random.choice(self.disjoint_clusters[k], size=C, replace=False))
                D.extend(np.random.choice(outer_diseases, size=n-C, replace=False))
            else:
                # adjusted C = C + (n - number of out-of-cluster diseases)
                D.extend(np.random.choice(self.disjoint_clusters[k], \
                    size = max(min(C, len(self.disjoint_clusters[k])), n+len(self.disjoint_clusters[k])-self.N), replace=False))
                D.extend(outer_diseases) # all the out-of-cluster diseases need to be added
        r[np.array(D)] = 1
        return list(r), n 
    
def run(file_name, dataset_size, alpha, s, num_diseases, num_clusters, tau, beta, p):
    n = np.zeros(num_diseases)
    with open(file_name, "w") as csvFile: 
        first_row = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        csv.writer(csvFile).writerow(first_row)
        for i in range(dataset_size):
            gen = DataGenerator(alpha=alpha, s=s, num_diseases=num_diseases, num_clusters=num_clusters, tau=tau, beta=beta, p=p)
            row, no_diseases = gen.generate_instance(False)
            n[no_diseases]+=1
            csv.writer(csvFile).writerow(row)
    csvFile.close()
    


if __name__ == '__main__':
    #example test case 
    run("../../dataset/synthetic_data_test_1.csv", 10000, 0.7, 1.5, 20, 5, [0.2,0.2,0.2,0.2,0.2], [0.2,0.2,0.2,0.2,0.2], 0.5)
