"""
Requirements:
mlxtend version = 0.15.0
python = 3.7.3
pyitlib= 0.2.2

Objective: Run the MaxEnt_MCC algorithm on the real data of 20 diseases. 

Sort the elements into order based on probabilities assigned to the individual vectors
and print the top 7 and bottom 7 elements
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
from operator import itemgetter
from collections import OrderedDict
import warnings 
warnings.simplefilter("ignore")

path_to_codebase = './codebase/'
sys.path.insert(0, path_to_codebase)
from codebase.utils import clean_preproc_data
from codebase.utils import clean_preproc_data_real
from codebase.extract_features import ExtractFeatures
from codebase.robust_optimizer_box import Optimizer as Optimizer_robust
from codebase.mba import marketbasket

def read_prob_dist(filename):
    with open(filename, "rb") as outfile:
        prob = pickle.load(outfile,encoding='latin1')
    return prob[1]

def read_model(filename):
    with open(filename,"rb") as outfile:
        sup_model = pickle.load(outfile)
    return sup_model

def compute_prob_exact(optobj):
    vecprob = dict()
    maxent_prob = []
    num_feats = optobj.feats_obj.data_arr.shape[1]
    
    maxent_sum_diseases = np.zeros(num_feats+1)
    all_perms = itertools.product([0, 1], repeat=num_feats)
    total_prob = 0.0    # finally should be very close to 1

    for tmp in all_perms:
        vec = np.asarray(tmp)
        p_vec = optobj.prob_dist(vec)
        vecprob[tmp] = p_vec
        # print('Vector:', vec, ' Probability: ', p_vec)
        j = sum(vec)
        maxent_sum_diseases[j] += p_vec
        total_prob += p_vec
        maxent_prob.append(p_vec) 
    
    print('Total Probability: ', total_prob)

    sorted_vecprob = OrderedDict(sorted(vecprob.items(), key=itemgetter(1), reverse=True))
    sorted_vecprob_a = OrderedDict(sorted(vecprob.items(), key=itemgetter(1)))
    print("Top 7 elements")
    [print(key,':',value) for key,value in list(sorted_vecprob.items())[:7]]
    print("Bottom 7 elements")
    [print(key,':',value) for key,value in list(sorted_vecprob_a.items())[:7]]

    # sorted_vecprob = {k:v for k,v in sorted(vecprob.items(), key=itemgetter(1))}

    emp_prob = np.zeros(num_feats + 1)
    for vec in optobj.feats_obj.data_arr:
        j = sum(vec)
        emp_prob[j] += 1
    emp_prob /= optobj.feats_obj.data_arr.shape[0]
    
    return maxent_prob, maxent_sum_diseases, emp_prob

def main(k=None):
    '''
    k: num of diseases
    '''
    # e = 0.64551868
    # d = 61166
    # k = 20
    # real_k = 1
    # # real_k = (k-4)/(20-4)
    # real_e = (e - 0.8)/(2.4-0.8)
    # # real_d = (d - 1000)/(60000-1000)
    # real_d = 1
    
    # # Random Forest Regression
    # sup_model = read_model('../support_analysis/rfregr_model.pickle')
    # covariates = np.array([real_k, real_e, real_d]).reshape(1,-1)
    # support = sup_model.predict(covariates)[0]

    #Measure time to compute maxent
    tic = time.time()

    directory = '../data/basket_sets_top20.csv'
    
    cleaneddata = clean_preproc_data_real(directory)
    size = cleaneddata.shape[0]
    support = 30/size
    print("The support is: ", support)

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

    print("\nThe constraints and corresponding supports are:")
    pair_constraints = dict()
    for two in two_wayc:
        pair_constraints[two] = round(support_dict[two],4)
    triple_constraints = dict()
    for three in three_wayc:
        triple_constraints[three] = round(support_dict[three],4)
    quad_constraints = dict()
    for four in four_wayc:
        quad_constraints[four] = round(support_dict[four],4)

    sorted_two = dict(sorted(pair_constraints.items(), key=itemgetter(1), reverse=True))
    sorted_three = dict(sorted(triple_constraints.items(), key=itemgetter(1), reverse=True))
    sorted_four = dict(sorted(quad_constraints.items(), key=itemgetter(1), reverse=True))
    
    print('Two way constraints:')
    print(sorted_two)
    print('Three way constraints:')
    print(sorted_three)
    print('Four way constraints:')
    print(sorted_four)

    # print('two_wayc', two_wayc.keys())
    # print('three_wayc', three_wayc.keys())
    # print('four_wayc', four_wayc.keys())
    
    print('The total number of MBA constraints are: ', str(len(two_wayc) + len(three_wayc) + len(four_wayc)))
    print()

    width = 1
    opt = Optimizer_robust(feats, 1) 

    soln_opt = opt.solver_optimize()
    if soln_opt == None:
        print('Solution does not converge')
        return 

    maxent, sum_prob_maxent, emp_prob = compute_prob_exact(opt)
    print()
    print("Empirical: " +str(emp_prob))
    print("Maxent: " + str(sum_prob_maxent))
   
    outfilename = '../data/maxent_mcc_30.pickle'
    
    with open(outfilename, "wb") as outfile:
        pickle.dump((maxent, sum_prob_maxent, emp_prob), outfile)
    
    toc = time.time()
    print('Computational time for calculating maxent = {} seconds'.format(toc-tic))

if __name__ == '__main__':
    k = 20
    main(k)