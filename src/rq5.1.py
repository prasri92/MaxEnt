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
from codebase.utils import clean_preproc_data_real
from codebase.extract_features import ExtractFeatures
from codebase.robust_optimizer_box import Optimizer as Optimizer_robust
from codebase.optimizer import Optimizer
#zero detection code implemented
# from codebase.optimizer_zeros import Optimizer as Optimizer_zeros
from codebase.mba import marketbasket

def read_prob_dist(filename):
    with open(filename, "rb") as outfile:
        prob = pickle.load(outfile,encoding='latin1')
    return prob[1]

def compute_prob_exact(optobj):
    maxent_prob = []
    num_feats = optobj.feats_obj.data_arr.shape[1]

    # print(optobj.feats_obj.feat_partitions)
    
    maxent_sum_diseases = np.zeros(num_feats+1)
    all_perms = itertools.product([0, 1], repeat=num_feats)
    total_prob = 0.0    # finally should be very close to 1

    for tmp in all_perms:
        vec = np.asarray(tmp)
        p_vec = optobj.prob_dist(vec)
        j = sum(vec)
        maxent_sum_diseases[j] += p_vec
        total_prob += p_vec
        maxent_prob.append(p_vec) 
    
    print("Maxent prob: ", maxent_sum_diseases)
    print('Total Probability: ', total_prob)
    print() 

    emp_prob = np.zeros(num_feats + 1)
    for vec in optobj.feats_obj.data_arr:
        j = sum(vec)
        emp_prob[j] += 1
    emp_prob /= optobj.feats_obj.data_arr.shape[0]
    
    return maxent_prob, maxent_sum_diseases, emp_prob

def main(file_num=None, k=None, dataset_num=None, support=None, i=None):
    '''
    file_num: handler for reference to different lambda's
    k: num of diseases
    support: input different support values
    '''
    #Measure time to compute maxent
    tic = time.time()
    print("#####################################################################")
    print("DATASET: ", dataset_num, " DISEASES: ", k, " FILE NUMBER: ", file_num, " SUPPORT: ", support)
    print("#####################################################################")
    print() 
    # generating synthetic data 
    directory = '../data/dataset_s'+str(dataset_num)+'/d'+str(k)+'/synthetic_data_expt'+str(file_num)+'.csv'
    
    # for synthetic data 
    cleaneddata = clean_preproc_data(directory)
    # for real data 
    # cleaneddata = clean_preproc_data_real(directory)

    support_dict, two_wayc, three_wayc, four_wayc = marketbasket(cleaneddata, support)
    if len(support_dict)==0 and len(two_wayc)==0 and len(three_wayc)==0 and len(four_wayc)==0:
        print("No associations are available with specified value of support, only MLE constraints")
        total_constraints = 0
    else:
        total_constraints = len(two_wayc)+len(three_wayc)+len(four_wayc)
    
    # if total_constraints > 250:
    # for d10, even over a 100, does not converge
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

    # print("\nThe constraints are:")
    # print('two_wayc', two_wayc)
    # print('three_wayc', three_wayc)
    # print('four_wayc', four_wayc)
    # print('The total number of constraints are: ', str(len(two_wayc) + len(three_wayc) + len(four_wayc)))
    num_constraints = len(two_wayc) + len(three_wayc) + len(four_wayc) 
    print("The number of MBA constraints are: \n", num_constraints)

    # opt = Optimizer(feats) 
    # width_factor = 1, width = size of the dataset 
    width = 1
    opt = Optimizer_robust(feats, width)

    #Use LP to detect zero atoms 
    # opt.exact_zero_detection(cleaneddata)
    # opt.approximate_zero_detection(cleaneddata)
    
    soln_opt = opt.solver_optimize()
    if soln_opt == None:
        print('Solution does not converge')
        return 

    maxent, sum_prob_maxent, emp_prob = compute_prob_exact(opt)
    print()
   
    # for synthetic data 
    outfilename = '../output/output_s'+str(dataset_num)+'/d'+str(k)+'_expt1.2/syn_maxent_expt'+str(file_num)+'_s'+str(i)+'.pickle'

    with open(outfilename, "wb") as outfile:
        pickle.dump((maxent, sum_prob_maxent, emp_prob, num_constraints, support), outfile)
    
    toc = time.time()
    print('Computational time for calculating maxent = {} seconds'.format(toc-tic))
    print() 

if __name__ == '__main__':
    # for synthetic data 
    num_dis = int(sys.argv[1])
    dataset_num = int(sys.argv[2])
    support_num = int(sys.argv[3])
    support = float(sys.argv[4])
    for j in range(1,11):
        main(file_num=j, k=int(num_dis), dataset_num=dataset_num, support=support, i=support_num)
        