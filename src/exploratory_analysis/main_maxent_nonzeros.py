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
from codebase.utils import clean_prepoc_data_nonzeros
from codebase.extract_features import ExtractFeatures
from codebase.optimizer_nonzeros import Optimizer
from codebase.mba import marketbasket

def read_prob_dist(filename):
    with open(filename, "rb") as outfile:
        prob = pickle.load(outfile,encoding='latin1')
    return prob[1]

def compute_prob_exact(optobj, prob_zeros, size):
    maxent_prob = []
    num_feats = optobj.feats_obj.data_arr.shape[1]
    maxent_diseases = np.zeros(num_feats+1)
    all_perms = list(itertools.product([0, 1], repeat=num_feats))

    total_prob = 0.0    # finally should be very close to 1

    print("Probability of zeros = ", prob_zeros)
    maxent_prob.append(prob_zeros)
    total_prob += prob_zeros
    maxent_diseases[0] = prob_zeros

    for tmp in all_perms[1:]:
        vec = np.asarray(tmp)
        p_vec = optobj.prob_dist(vec)*(1-prob_zeros)
        # print('Vector:', vec, ' Probability: ', p_vec)
        j = sum(vec)
        maxent_diseases[j] += p_vec 
        total_prob += p_vec
        maxent_prob.append(p_vec) 
    """
    disease1and3 = {'00':0, '01':0, '10':0, '11':0}

    all_perms = itertools.product([0, 1], repeat=num_feats)
    #Compute marginals for D1 and D3 to check
    for tmp in all_perms:
        vec = np.asarray(tmp)
        p_vec = optobj.prob_dist(vec)
        s = str(vec[1]) + str(vec[3])
        disease1and3[s] += p_vec

    print('Disease 1 and 3: ', disease1and3)
    """
    print("Total Probability: " + str(total_prob))

    emp_prob = np.zeros(num_feats + 1)
    emp_prob[0] = prob_zeros
    for vec in optobj.feats_obj.data_arr:
        j = sum(vec)
        emp_prob[j] += 1
    # emp_prob /= optobj.feats_obj.data_arr.shape[0]
    emp_prob[1:] /= int(size)
    
    return maxent_prob, maxent_diseases, emp_prob

def main(file_num=None, size=None, support=None, trial=None):
    print("File num: " + str(file_num) + " has started\n")
    
    #Determining support for market basket analysis
    # support_data_overlap_nonzero = {1:0.003, 2:0.002, 3:0.002, 4:0.001, 5:0.002, 6:0.007, \
    #     7:0.008, 8:0.01, 9:0.006, 10:0.07, 11:0.016, 12:0.015, 13:0.016, \
    #     14:0.014, 15:0.014, 16:0.028, 17:0.028, 18:0.02, 19:0.02, 20:0.02, 21:0.035, \
    #     22:0.04, 23:0.041, 24:0.04, 25:0.035}  
    # support = support_data_overlap_nonzero[file_num]

    # for four diseases
    # support = 0.02

    # for ten diseases
    support_dict = {3:0.029, 13:0.155, 23:0.18}
    support = support_dict[file_num]
 
    #Measuring time for MaxEnt computation
    tic = time.time()

    # real data
    # directory = '../dataset/basket_sets.csv'
    # generating synthetic data 
    directory = '../dataset/d'+str(size)+'_4/synthetic_data_expt'+str(file_num)+'.csv'

    cleaneddata, prob_zeros = clean_prepoc_data_nonzeros(directory)

    # two_wayc, three_wayc, four_wayc = marketbasket(cleaneddata, support)
    # two_wayc = {}
    # two_wayc = {(1, 3): (1, 1)} 

    # two_wayc = {(1, 0): (1, 1), (0, 3): (1, 1)} 
    two_wayc = {(1, 0): (1, 1), (0, 3): (1, 1) , (3, 2): (1, 1), (0, 2): (1, 1)}
    # two_wayc = {(1, 0): (1, 1), (0, 3): (1, 1), (3, 2): (1, 1), (0, 2): (1, 1), (1, 3): (1, 1), (1, 2): (1, 1)}

    # three_wayc = {}
    three_wayc = {(1, 0, 3): (1, 1, 1), (0, 3, 2): (1, 1, 1)}  
    # three_wayc = {(1, 0, 3): (1, 1, 1), (0, 3, 2): (1, 1, 1), (1, 3, 2): (1, 1, 1), (1, 0, 2): (1, 1, 1)}
    
    # four_wayc = {}
    four_wayc = {(1, 0, 3, 2): (1, 1, 1, 1)}
    
    feats = ExtractFeatures(cleaneddata.values)

    feats.set_two_way_constraints(two_wayc)
    feats.set_three_way_constraints(three_wayc)
    feats.set_four_way_constraints(four_wayc)

    feats.partition_features()
    print(feats.feat_partitions)

    print("\nThe constraints are:")
    print('two_wayc', two_wayc)
    print('three_wayc', three_wayc)
    print('four_wayc', four_wayc)
    print()

    opt = Optimizer(feats) 
    opt.exact_zero_detection(cleaneddata)
    # opt.approximate_zero_detection(cleaneddata)

    soln_opt = opt.solver_optimize()
    print("Optimizer is done. Computing probabilities")

    maxent, sum_prob_maxent, emp_prob = compute_prob_exact(opt, prob_zeros, size)
    print("Empirical: " +str(emp_prob))
    print("Maxent: " + str(sum_prob_maxent))
    print("True distribution:" + str(read_prob_dist('../output/d'+str(size)+'_4/truedist_expt'+str(file_num)+'.pickle')))
    

    # for real data'support_'+str(support)+
    # outfilename = '../output/realdata_maxent.pickle'
    # for synthetic data 
    # outfilename = '../output/d'+str(size)+'_4_nonzeros/syn_maxent_expt'+str(file_num)+'.pickle'#'_constraints_'+str(trial)+'.pickle' #

    # with open(outfilename, "wb") as outfile:
    #     pickle.dump((maxent, sum_prob_maxent, emp_prob), outfile)

    toc = time.time()
    print('Computational time for calculating maxent = {} seconds'.format(toc-tic))

if __name__ == '__main__':
    file_num = sys.argv[1]
    size = sys.argv[2]
    main(file_num=int(file_num), size=size)
