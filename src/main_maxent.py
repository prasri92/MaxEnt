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
from codebase.optimizer import Optimizer
from codebase.mba import marketbasket

def read_prob_dist(filename):
    with open(filename, "rb") as outfile:
        prob = pickle.load(outfile,encoding='latin1')
    return prob[1]

def compute_prob_exact(optobj):
    maxent_prob = []
    num_feats = optobj.feats_obj.data_arr.shape[1]

    print(optobj.feats_obj.feat_partitions)
    
    maxent_sum_diseases = np.zeros(num_feats+1)
    all_perms = itertools.product([0, 1], repeat=num_feats)
    total_prob = 0.0    # finally should be very close to 1

    for tmp in all_perms:
        vec = np.asarray(tmp)
        p_vec = optobj.prob_dist(vec)
        # print('Vector:', vec, ' Probability: ', p_vec)
        j = sum(vec)
        maxent_sum_diseases[j] += p_vec
        total_prob += p_vec
        maxent_prob.append(p_vec) 
    
    print('Total Probability: ', total_prob)

    emp_prob = np.zeros(num_feats + 1)
    for vec in optobj.feats_obj.data_arr:
        j = sum(vec)
        emp_prob[j] += 1
    emp_prob /= optobj.feats_obj.data_arr.shape[0]
    
    return maxent_prob, maxent_sum_diseases, emp_prob

def main(file_num=None):
    #Support for marketbasket analysis
    # for 20 diseases
    sups = {1:0.002, 2:0.002, 3:0.002, 4:0.002, 5:0.002, 6:0.004, \
        7:0.005, 8:0.004, 9:0.004, 10:0.004, 11:0.012, 12:0.009, 13:0.01, \
        14:0.01, 15:0.01, 16:0.023, 17:0.022, 18:0.018, 19:0.014, 20:0.014, 21:0.026, \
        22:0.032, 23:0.034, 24:0.029, 25:0.03}
    support = sups[file_num]
    
    # for four diseases
    # sups = {3:0.02 , 13:0.08, 23:0.12}
    # support = sups[file_num]
    
    # for ten diseases
    # sups = {3:0.001, 13:0.09, 23:0.14}
    # sups = {3:0.001, 13:0.058, 23:0.085}
    # support = sups[file_num]
    
    #Measure time to compute maxent
    tic = time.time()

    # real data
    # directory = '../dataset/basket_sets.csv'
    # generating synthetic data 
    # directory = '../dataset/d50_4/synthetic_data_expt'+str(file_num)+'.csv'
    # directory = '../dataset/d250_10/synthetic_data_expt'+str(file_num)+'.csv'
    directory = '../dataset/d500_20/synthetic_data_expt'+str(file_num)+'.csv'
    
    cleaneddata = clean_preproc_data(directory)

    support_dict, two_wayc, three_wayc, four_wayc = marketbasket(cleaneddata, support)
    # two_wayc = {(1,3):(1,1)}
    # three_wayc = {}
    # four_wayc = {}

    # feats = ExtractFeatures(cleaneddata.values)
    feats = ExtractFeatures(cleaneddata.values, Mu=10)

    feats.set_two_way_constraints(two_wayc)
    feats.set_three_way_constraints(three_wayc)
    feats.set_four_way_constraints(four_wayc)
    feats.set_supports(support_dict)

    feats.partition_features()
    print("The approximated clusters are:\n", feats.feat_partitions)


    print("\nThe constraints are:")
    print('two_wayc', two_wayc)
    print('three_wayc', three_wayc)
    print('four_wayc', four_wayc)
    print()

    print(feats.feat_partitions)
    opt = Optimizer(feats) 
    #Use LP to detect zero atoms 
    # opt.exact_zero_detection(cleaneddata)
    # opt.approximate_zero_detection(cleaneddata)
    
    soln_opt = opt.solver_optimize()
    if soln_opt == None:
        print('Solution does not converge')
        return 

    maxent, sum_prob_maxent, emp_prob = compute_prob_exact(opt)
    print()
    print("Empirical: " +str(emp_prob))
    print("Maxent: " + str(sum_prob_maxent))
    # print("True distribution:" + str(read_prob_dist('../output/d50_4/truedist_expt'+str(file_num)+'.pickle')))
    # print("True distribution:" + str(read_prob_dist('../output/d250_10/truedist_expt'+str(file_num)+'.pickle')))
    print("True distribution:" + str(read_prob_dist('../output/d500_20/truedist_expt'+str(file_num)+'.pickle')))
    
    # for real data
    # outfilename = '../output/realdata_maxent.pickle'
    # for synthetic data 
    # outfilename = '../output/d50_4/syn_maxent_expt'+str(file_num)+'.pickle'
    # outfilename = '../output/d250_10/syn_maxent_expt'+str(file_num)+'.pickle'
    outfilename = '../output/d500_20/syn_maxent_expt'+str(file_num)+'.pickle'

    with open(outfilename, "wb") as outfile:
        pickle.dump((maxent, sum_prob_maxent, emp_prob), outfile)
    
    toc = time.time()
    print('Computational time for calculating maxent = {} seconds'.format(toc-tic))

if __name__ == '__main__':
    file_num = sys.argv[1]
    main(int(file_num))
