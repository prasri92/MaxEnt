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
from codebase.robust_optimizer_exp import Optimizer as Optimizer_exp

# for box constraints
from codebase.robust_optimizer_box import Optimizer as Optimizer_box

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


def calc_maxent(file_num, directory, perturb_prob, k, width=None):
    """
    Calculate maximum entropy for each perturbation and width and return results
    """
    support_vals_3 = {4:{1:0.001, 2:0.001, 3:0.001, 4:0.001, 5:0.001}, 
                7:{1:0.005, 2:0.02, 3:0.05, 4:0.05, 5:0.05}, 
                10:{1:0.005, 2:0.005, 3:0.02, 4:0.04, 5:0.04},
                15:{1:0.001, 2:0.005, 3:0.01, 4:0.02, 5:0.04}}
    support = support_vals_3[k][file_num]
    
    #Measure time to compute maxent
    tic = time.time()
    
    cleaneddata = clean_preproc_data_perturb(directory, perturb_prob)

    support_dict, two_wayc, three_wayc, four_wayc = marketbasket(cleaneddata, support)

    if len(two_wayc) + len(three_wayc) + len(four_wayc) > 100:
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
    lambdas = {1:0.42, 2:0.5, 3:0.62, 4:0.83, 5:1.25}
    opt_r = Optimizer_exp(feats, lambdas[file_num])


    # Use regularization methods of box constraints
    # width = {4:50, 7:225, 10:350, 15:350}
    # opt_r = Optimizer_box(feats, width[k]) 

    # for testing different widths 
    # opt_r = Optimizer_box(feats, width)
    
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

def main(file_num, k):
    '''
    Function to store multiple perturbation and width maximum entropy distributions in a single file
    '''
    # generating synthetic data 
    directory = '../dataset/d'+str(k)+'/synthetic_data_expt'+str(file_num)+'.csv'

    # for 10 diseases
    # widths = [50, 150, 250, 350, 450]

    # for 7 diseases 
    # widths = [25, 75, 125, 175, 225]

    # for 4 diseases
    # widths = [10,30,50,70,90]

    # IF testing for different widths (also change file name)
    # for w in width:

    perturb_prob = [0.01, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21]
    for p in perturb_prob:
        maxent_dis_p = []
        sum_dis_p = []
        maxent_dis_r = []
        sum_dis_r = []
        tup = calc_maxent(file_num, directory, perturb_prob=p, k=k)
        maxent_dis_p.append(tup[0]) 
        sum_dis_p.append(tup[1])
        maxent_dis_r.append(tup[2]) 
        sum_dis_r.append(tup[3])

        output = (maxent_dis_p, sum_dis_p, maxent_dis_r, sum_dis_r)

        # for synthetic data 
        outfilename = '../output/d'+str(k)+'/syn_maxent_perturb_box_expt'+str(file_num)+'_'+str(p)+'.pickle'

        with open(outfilename, "wb") as outfile:
            pickle.dump(output, outfile)

if __name__ == '__main__':
    # for synthetic data 
    num_dis = sys.argv[1]
    file_num = sys.argv[2]
    main(file_num=int(file_num), k=int(num_dis))
