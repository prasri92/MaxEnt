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
from codebase.utils import clean_preproc_data_perturb
from codebase.extract_features import ExtractFeatures
from codebase.mba import marketbasket
from codebase.optimizer import Optimizer

# for exponential prior contstraints
from codebase.robust_optimizer import Optimizer as Optimizer_exp

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
    return maxent_prob, maxent_sum_diseases

def calc_emp_prob(optobj):

    num_feats = optobj.feats_obj.data_arr.shape[1]
    emp_prob = np.zeros(num_feats + 1)
    for vec in optobj.feats_obj.data_arr:
        j = sum(vec)
        emp_prob[j] += 1
    emp_prob /= optobj.feats_obj.data_arr.shape[0]
    
    return emp_prob


def calc_maxent(file_num, directory, perturb_prob=None):
    """
    Calculate maximum entropy for each perturbation and width and return results
    """
    #Support for marketbasket analysis
    # for 20 diseases
    # sups = {1:0.002, 2:0.002, 3:0.002, 4:0.002, 5:0.002, 6:0.004, \
    #     7:0.005, 8:0.004, 9:0.004, 10:0.004, 11:0.012, 12:0.009, 13:0.01, \
    #     14:0.01, 15:0.01, 16:0.023, 17:0.022, 18:0.018, 19:0.014, 20:0.014, 21:0.026, \
    #     22:0.032, 23:0.04, 24:0.029, 25:0.03}
    # sups = {3:0.002, 13: 0.01, 23: 0.06}
    
    # for four diseases
    # sups = {3:0.02 , 13:0.08, 23:0.12}
    
    # for ten diseases
    # sups = {3:0.001, 13:0.09, 23:0.14}
    # sups = {3:0.001, 13:0.058, 23:0.085} #constraints are too many
    # sups = {3:0.001, 13:0.05, 23:0.07} #constraints are too many

    # for 15 diseases
    sups = {3: 0.002, 13: 0.02, 23: 0.06}

    support = sups[file_num]
    
    #Measure time to compute maxent
    tic = time.time()
    
    cleaneddata = clean_preproc_data_perturb(directory, perturb_prob)

    support_dict, two_wayc, three_wayc, four_wayc = marketbasket(cleaneddata, support)

    if len(two_wayc) + len(three_wayc) + len(four_wayc) > 180:
        feats = ExtractFeatures(cleaneddata.values, Mu=7)
    else:
        feats = ExtractFeatures(cleaneddata.values)
    
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
    print('The total number of constraints are: ', str(len(two_wayc) + len(three_wayc) + len(four_wayc)))
    print()

    print(feats.feat_partitions)

    #Use LP to detect zero atoms 
    # opt.exact_zero_detection(cleaneddata)
    # opt.approximate_zero_detection(cleaneddata)
    
    opt_p = Optimizer(feats)
    soln_opt_p = opt_p.solver_optimize()
    if soln_opt_p == None:
        print('Solution does not converge')
        return 

    maxent_p, sum_prob_maxent_p = compute_prob_exact(opt_p)

    print("Maxent Prob: (Perturbed)", sum_prob_maxent_p)
    print

    # Use regularization methods of exponential prior 
    variance = {3:0.64, 13:5.778476, 23:16.0}
    opt_r = Optimizer_exp(feats, variance[file_num]) 
    soln_opt_r = opt_r.solver_optimize()
    if soln_opt_r == None:
        print('Solution does not converge')
        return 

    maxent_r, sum_prob_maxent_r = compute_prob_exact(opt_r)

    print("Maxent Prob: ", sum_prob_maxent_r)
    print

    toc = time.time()
    print('Computational time for calculating maxent = {} seconds'.format(toc-tic))

    return (maxent_p, sum_prob_maxent_p, maxent_r, sum_prob_maxent_r)

def main(file_num):
    '''
    Function to store multiple perturbation and width maximum entropy distributions in a single file
    '''
    # real data
    # directory = '../dataset/basket_sets.csv'
    # generating synthetic data 
    # directory = '../dataset/d50_4/synthetic_data_expt'+str(file_num)+'.csv'
    # directory = '../dataset/d200_4/synthetic_data_expt'+str(file_num)+'.csv'
    # directory = '../dataset/d250_10/synthetic_data_expt'+str(file_num)+'.csv'
    directory = '../dataset/d350_15/synthetic_data_expt'+str(file_num)+'.csv'
    # directory = '../dataset/d500_20/synthetic_data_expt'+str(file_num)+'.csv'

    perturb_prob = [0.01, 0.04, 0.1, 0.2]
    i = 0
    for p in perturb_prob:
        maxent_dis_p = []
        sum_dis_p = []
        maxent_dis_r = []
        sum_dis_r = []
        tup = calc_maxent(file_num, directory, perturb_prob=p)
        maxent_dis_p.append(tup[0]) 
        sum_dis_p.append(tup[1])
        maxent_dis_r.append(tup[2]) 
        sum_dis_r.append(tup[3])

        output = (maxent_dis_p, sum_dis_p, maxent_dis_r, sum_dis_r)

        # for real data
        # outfilename = '../output/realdata_maxent.pickle'
        # for synthetic data 
        # outfilename = '../output/d50_4_perturb/syn_maxent_expt'+str(file_num)+'_'+str(p)+'.pickle'
        # outfilename = '../output/d200_4_perturb/syn_maxent_expt'+str(file_num)+'_'+str(p)+'.pickle'
        # outfilename = '../output/d250_10_perturb/syn_maxent_expt'+str(file_num)+'_'+str(p)+'.pickle'
        outfilename = '../output/d350_15_perturb/syn_maxent_expt'+str(file_num)+'_'+str(p)+'.pickle'
        # outfilename = '../output/d500_20_perturb/syn_maxent_expt'+str(file_num)+'_'+str(p)+'.pickle'

        with open(outfilename, "wb") as outfile:
            pickle.dump(output, outfile)

if __name__ == '__main__':
    file_num = sys.argv[1]
    main(int(file_num))
