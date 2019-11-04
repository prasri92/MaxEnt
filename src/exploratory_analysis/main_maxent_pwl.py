"""
mlxtend version = 0.15.0
python = 3.7.3
"""
from __future__ import division
import pickle
import numpy as np
import itertools
from matplotlib import pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import sys
import time

path_to_codebase = './codebase/'
sys.path.insert(0, path_to_codebase)
from codebase.utils import clean_preproc_data
from codebase.extract_features import ExtractFeatures
from codebase.optimizer_pwl import Optimizer
from codebase.mba import marketbasket


class MaxEnt(object):
    """
    Class Summary:
    Produces an output for maximum entropy model by feeding in the synthetic and real data files. 
    Reads in the true distribution and compares to the synthetic distribution and writes to file. 
    n represents the total number of diseases
    Attributes
    ----------
    maxent_prob: Numpy array of size 2**n containing each p(r) that we are trying to find in the 
    piecewise likelihood method
    maxent_sum_prob: Numpy array of size n+1 containing the sum of probabilities for patients with x number 
    of diseases
    emp_prob: Numpy array containing the empirical probabilities for the # of diseases in the dataset. 
    """

    def __init__(self):
        self.num_feats = None
        self.maxent_prob = None
        self.maxent_sum_diseases = None
        self.total_prob = 0.0

    def set_probs(self, optobj):
        self.num_feats = optobj.feats_obj.data_arr.shape[1]
        self.maxent_prob = np.ones(2**self.num_feats)
        self.maxent_sum_diseases = np.zeros(self.num_feats+1)

    def read_prob_dist(self, filename):
        '''
        Reads the true probabilities from the file
        '''
        with open(filename, "rb") as outfile:
            prob = pickle.load(outfile,encoding='latin1')
        return prob[1]

    def compute_prob_pwl(self, optobj):
        '''
        Compute probabilities using the piecewise likelihood method
        '''
        all_perms = itertools.product([0,1], repeat=self.num_feats)

        for i, vec in enumerate(all_perms):
            vec = np.asarray(vec)
            j = sum(vec)
            self.maxent_sum_diseases[j] += self.maxent_prob[i]


    def main(self, file_num=None):
        #Support for marketbasket analysis
        # for 20 diseases
        # sups = {1:0.002, 2:0.002, 3:0.002, 4:0.002, 5:0.002, 6:0.004, \
        #     7:0.005, 8:0.004, 9:0.004, 10:0.004, 11:0.012, 12:0.009, 13:0.01, \
        #     14:0.01, 15:0.01, 16:0.023, 17:0.022, 18:0.018, 19:0.014, 20:0.014, 21:0.026, \
        #     22:0.032, 23:0.04, 24:0.029, 25:0.03}
        # support = sups[file_num]
        
        # for four diseases
        # sups = {3:0.02 , 13:0.08, 23:0.12}
        # support = sups[file_num]
        # 
        # for ten diseases
        # sups = {3:0.001, 13:0.09, 23:0.14}
        # sups = {3:0.001, 13:0.058, 23:0.085}
        sups = {3:0.001, 13:0.03, 23:0.085}
        # sups = {3:0.008}
        support = sups[file_num]
        
        #Measure time to compute maxent
        tic = time.time()

        # real data
        # directory = '../dataset/basket_sets.csv'
        # generating synthetic data 
        directory = '../dataset/d50_4/synthetic_data_expt'+str(file_num)+'.csv'
        # directory = '../dataset/d200_4/synthetic_data_expt'+str(file_num)+'.csv'
        # directory = '../dataset/d250_10/synthetic_data_expt'+str(file_num)+'.csv'
        # directory = '../dataset/d500_20/synthetic_data_expt'+str(file_num)+'.csv'
        
        cleaneddata = clean_preproc_data(directory)

        support_dict, two_wayc, three_wayc, four_wayc = marketbasket(cleaneddata, support)  
        # two_wayc = {(0,1):(1,1), (3,5):(1,1)}
        # three_wayc = {}
        # four_wayc = {}

        feats = ExtractFeatures(cleaneddata.values)
        feats.set_two_way_constraints(two_wayc)
        feats.set_three_way_constraints(three_wayc)
        feats.set_four_way_constraints(four_wayc)

        feats.partition_features()
        print("The approximated clusters before piecewise partitioning are:\n", feats.feat_partitions)

        opt = Optimizer(feats)
        opt.util_compute_fcs(feats.feat_partitions[0])
        soln_opt = opt.solver_optimize()
        print("Solution is:", soln_opt)

        self.set_probs(opt)
        self.maxent_prob = opt.compute_all_prob(self.num_feats)

        self.compute_prob_pwl(opt)
        print('MaxEnt', self.maxent_sum_diseases)
        
        # print("Empirical: " +str(emp_prob))
        # print("Maxent: " + str(sum_prob_maxent))
        # print("True distribution:" + str(self.read_prob_dist('../output/d250_10/truedist_expt'+str(file_num)+'.pickle')))

        # outfilename = '../output/d250_10/syn_maxent_expt'+str(file_num)+'.pickle'
        # with open(outfilename, "wb") as outfile:
            # pickle.dump((maxent, sum_prob_maxent, emp_prob), outfile)
        
        toc = time.time()
        print('Computational time for calculating maxent = {} seconds'.format(toc-tic))
       

file_num = sys.argv[1]
maxent_obj = MaxEnt()
maxent_obj.main(int(file_num))
