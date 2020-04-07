"""
Requirements:
mlxtend version = 0.15.0
python = 3.7.3

Objective: Run a basic maxent algorithm on the base dataset, with either chosen/learned support values to see the performance
of either the robust or non-robust maxent model. 

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

path_to_codebase = './codebase/'
sys.path.insert(0, path_to_codebase)
from codebase.utils import clean_preproc_data
from codebase.utils import clean_preproc_data_real
from codebase.extract_features import ExtractFeatures
# from codebase.robust_optimizer_box import Optimizer as Optimizer_robust
from codebase.optimizer import Optimizer
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

    # print(optobj.feats_obj.feat_partitions)
    
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

def main(file_num=None, k=None, support=None, dataset_num=None):
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

    # exp = {1:0.8 , 2:1.2 , 3:1.6 , 4:2.0 , 5:2.4}
    # size = {4: 50, 7: 125, 10: 250, 15:350}
    # e = exp[file_num]
    # d_size = size[k]
    # real_k = (k-4)/(20-4)
    # real_e = (e - 0.8)/(2.4-0.8)
    # real_d = (d_size - 51)/(1000-51)
    
    # # Random Forest Regression
    # sup_model = read_model('../support_analysis/rfregr_model.pickle')
    # covariates = np.array([real_k, real_e, real_d]).reshape(1,-1)
    # support = sup_model.predict(covariates)[0]

    # Polynomial Regression deg = 5
    # sup_model = read_model('../support_analysis/preg5_model.pickle')
    # covariates = np.array([real_k, real_e, real_d]).reshape(1,-1)
    # from sklearn.preprocessing import PolynomialFeatures 
    # poly = PolynomialFeatures(degree = 5) 
    # support = sup_model.predict(poly.fit_transform(covariates))[0]

    print("The support is: ", support)

    #Measure time to compute maxent
    tic = time.time()

    # generating synthetic data 
    directory = '../dataset/d'+str(k)+'/synthetic_data_expt'+str(file_num)+'.csv'
    # alternate use on cluster
    # directory = '../data/dataset_s'+str(dataset_num)+'/d'+str(k)+'/synthetic_data_expt'+str(file_num)+'.csv'
    
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
    print('two_wayc', two_wayc)
    print('three_wayc', three_wayc)
    print('four_wayc', four_wayc)
    print('The total number of MBA constraints are: ', str(len(two_wayc) + len(three_wayc) + len(four_wayc)))
    print()

    # width = 1
    # opt = Optimizer_robust(feats, 1) 
    opt = Optimizer(feats)

    soln_opt = opt.solver_optimize()
    if soln_opt == None:
        print('Solution does not converge')
        return 

    maxent, sum_prob_maxent, emp_prob = compute_prob_exact(opt)
    print()
    print("Empirical: " +str(emp_prob))
    print("Maxent: " + str(sum_prob_maxent))
    #without dataset
    print("True distribution:" + str(read_prob_dist('../output/d'+str(k)+'/truedist_expt'+str(file_num)+'.pickle')))
    # with dataset 
    # print("True distribution:" + str(read_prob_dist('../output/output_s'+str(dataset_num)+'/d'+str(k)+'/truedist_expt'+str(file_num)+'.pickle')))
   
    # for synthetic data 
    outfilename = '../output/d'+str(k)+'_cs/syn_maxent_expt'+str(file_num)+'.pickle'
    
    with open(outfilename, "wb") as outfile:
        pickle.dump((maxent, sum_prob_maxent, emp_prob), outfile)
    
    toc = time.time()
    print('Computational time for calculating maxent = {} seconds'.format(toc-tic))

if __name__ == '__main__':
    # for synthetic data 
    
    # with dataset and learned support 
    # num_dis = sys.argv[1]
    # dataset_num = sys.argv[2]
    # file_num = sys.argv[3]
    # support = sys.argv[4]
    # main(file_num=int(file_num), k=int(num_dis), dataset_num=int(dataset_num))

    # without dataset
    num_dis = sys.argv[1]
    file_num = sys.argv[2]
    main(int(file_num), int(num_dis))