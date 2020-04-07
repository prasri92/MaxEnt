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
from collections import OrderedDict
from operator import itemgetter
import sys
import time
import csv

'''
Experiment to compare the probabilities assigned to zero vectors in three cases
1. with zero detection 
2. Without zero detection and without regularization
3. Without zero detection and with regularization 
'''

path_to_codebase = './codebase/'
sys.path.insert(0, path_to_codebase)
from codebase.utils import clean_preproc_data
from codebase.utils import clean_preproc_data_real
from codebase.extract_features import ExtractFeatures
from codebase.optimizer_zeros import Optimizer as Optimizer_zeros
from codebase.optimizer import Optimizer
from codebase.robust_optimizer_box import Optimizer as Optimizer_box
from codebase.mba import marketbasket

def read_prob_dist(filename):
    with open(filename, "rb") as outfile:
        prob = pickle.load(outfile,encoding='latin1')
    return prob[1]

def compute_prob(optobj, prob_vectors):
    ###################################################
    all_vec = {}
    for vec in prob_vectors:
        vector = np.asarray(vec)
        p_vector = optobj.prob_dist(vector)
        all_vec[tuple(vec)] = p_vector

    return all_vec
    ###################################################
    
def compute_zero_atoms(optobj):
    ###################################################
    num_feats = optobj.feats_obj.data_arr.shape[1]
    all_zeros = []
    
    all_perms = itertools.product([0, 1], repeat=num_feats)

    for tmp in all_perms:
        vec = np.asarray(tmp)
        p_vec = optobj.prob_dist(vec)
        if p_vec == 0.0:
            all_zeros.append(vec)
    
    number_zeros = len(all_zeros)
    print("Number of zero vectors in total: ", number_zeros)
    if number_zeros != 0:
        return all_zeros, True
    else:
        return all_zeros, False
    ###################################################

def main(file_num=None, k=None, dataset_num=None):
    '''
    file_num: handler for reference to different lambda's
    k: num of diseases
    support: input different support values
    '''
    #Support for marketbasket analysis
    support_vals = {3:{1:0.06, 2:0.06, 3:0.0751, 4:0.0792, 5:0.0751},
                    4:{1:0.072, 2:0.0763, 3:0.0751, 4:0.0792, 5:0.0751}, 
                    7:{1:0.06, 2:0.0375, 3:0.0452, 4:0.0686, 5:0.848},
                    10:{1:0.0632, 2:0.0298, 3:0.0267, 4:0.0378, 5:0.0545},
                    15:{1:0.0086, 2:0.0106, 3:0.0127, 4:0.0205, 5:0.0295}}
    support = support_vals[k][file_num]
    
    #Measure time to compute maxent
    tic = time.time()

    if dataset_num == None:
        # generating synthetic data 
        directory = '../dataset/d'+str(k)+'/synthetic_data_expt'+str(file_num)+'.csv'
    else:
        # with dataset number
        directory = '../dataset_s'+str(dataset_num)+'/d'+str(k)+'/synthetic_data_expt'+str(file_num)+'.csv'
    
    # for synthetic data 
    cleaneddata = clean_preproc_data(directory)
    # for real data 
    # cleaneddata = clean_preproc_data_real(directory)

    support_dict, two_wayc, three_wayc, four_wayc = marketbasket(cleaneddata, support)

    total_constraints = len(two_wayc)+len(three_wayc)+len(four_wayc)
    if total_constraints > 100:
        feats = ExtractFeatures(cleaneddata.values, Mu=7)
    else:
        feats = ExtractFeatures(cleaneddata.values)

    feats.set_two_way_constraints(two_wayc)
    feats.set_three_way_constraints(three_wayc)
    feats.set_four_way_constraints(four_wayc)
    feats.set_supports(support_dict)

    feats.partition_features()

    print("The clusters are:\n", feats.feat_partitions)

    print("\nThe constraints are:")
    print('Marginals',list(range(k)))
    print('Two_wayc', two_wayc)
    print('Three_wayc', three_wayc)
    print('Four_wayc', four_wayc)
    print('The total number of constraints are: ', str(len(two_wayc) + len(three_wayc) + len(four_wayc) + k))
    print()

    opt_z = Optimizer_zeros(feats) 
    #Use LP to detect zero atoms 
    opt_z.exact_zero_detection(cleaneddata)

    soln_opt = opt_z.solver_optimize()
    if soln_opt == None:
        print('Solution does not converge')
        return 

    zero_atoms, z_flag = compute_zero_atoms(opt_z)

    if z_flag == True:
        opt = Optimizer(feats)
        soln_opt = opt.solver_optimize()
        if soln_opt == None:
            print('Solution does not converge')
            return 

        zero_probs = compute_prob(opt, zero_atoms)

        width = {4:50, 7:125, 10:250, 15:350}
        opt_r = Optimizer_box(feats, width[k])
        soln_opt = opt_r.solver_optimize()
        if soln_opt == None:
            print('Solution does not converge')
            return 

        zero_probs_r = compute_prob(opt_r, zero_atoms)
        
        print("Zero atoms in three cases\n")
        z = pd.Series(zero_probs).to_frame('prob_non_robust')
        z_r = pd.Series(zero_probs_r).to_frame('prob_robust')
        full_z = pd.concat([z, z_r], axis=1)
        
        if dataset_num == None:
            outfilename = 'outfiles/rq7.1/d'+str(k)+'_f'+str(file_num)+'.csv'
        else:
            outfilename = 'outfiles/rq7.1/ds'+str(dataset_num)+'_d'+str(k)+'_f'+str(file_num)+'.csv'
        full_z.to_csv(outfilename, index=False)    
    
    else:
        print("No zeros!")

    toc = time.time()
    time_taken = toc-tic
    print('Computational time for calculating maxent = {} seconds'.format(toc-tic))
    # return tot_zeros, zv_nzemp, zv_zemp, nzv_zemp, time_taken

if __name__ == '__main__':
    num_dis = sys.argv[1]
    file_num = sys.argv[2]
    if len(sys.argv) > 3:
        dataset_num = sys.argv[3]
        main(k=int(num_dis), file_num=int(file_num), dataset_num=int(dataset_num))
    else:
        main(k=int(num_dis), file_num=int(file_num))