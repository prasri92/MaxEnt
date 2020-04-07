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
from codebase.utils import clean_preproc_data_perturb, clean_preproc_data
from codebase.extract_features import ExtractFeatures
from codebase.mba import marketbasket
from codebase.optimizer import Optimizer
# from codebase.optimizer_zeros import Optimizer

# for exponential prior contstraints
from codebase.robust_optimizer_exp import Optimizer as Optimizer_exp

# for box constraints
from codebase.robust_optimizer_box_ci import Optimizer as Optimizer_box

def read_prob_dist(filename):
    with open(filename, "rb") as outfile:
        prob = pickle.load(outfile,encoding='latin1')
    return prob[1]

def read_model(filename):
    with open(filename,"rb") as outfile:
        sup_model = pickle.load(outfile)
    return sup_model

def compute_prob_exact(optobj, k, fnum, p):
    maxent_prob = []
    num_feats = optobj.feats_obj.data_arr.shape[1]

    print(optobj.feats_obj.feat_partitions)
    
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
    
    print('Total Probability: ', total_prob)
    zeros = np.count_nonzero(maxent_prob==0.0)
    # if zeros > 0:
    #     print("There are zeros: ", zeros)
    # else:
    #     print("NO ZEROS!")
    write_file = 'outfiles/rq2.3/zeros_check.txt'
    with open(write_file, 'a') as f: 
        line = 'Diseases: '+str(k)+' File number: '+ str(fnum)+' Pert: '+str(p)+' Zeros: '+str(zeros)+'\n'
        f.write(line)
    f.close()
    return maxent_prob, maxent_sum_diseases

def calc_emp_prob(optobj):

    num_feats = optobj.feats_obj.data_arr.shape[1]
    emp_prob = np.zeros(num_feats + 1)
    for vec in optobj.feats_obj.data_arr:
        j = sum(vec)
        emp_prob[j] += 1
    emp_prob /= optobj.feats_obj.data_arr.shape[0]
    
    return emp_prob

def calc_maxent_unperturbed(file_num, directory, k, width=None):
    """
    Calculate maximum entropy for each perturbation and width and return results
    """
    #tested for lambda = 2/3
    support_vals = {4:{1:0.0792, 2:0.063, 3:0.0671, 4:0.0799, 5:0.0671}, 
                    7:{1:0.0725, 2:0.0558, 3:0.0529, 4:0.0661, 5:0.0835},
                    10:{1:0.083, 2:0.0555, 3:0.0496, 4:0.0414, 5:0.0571},
                    15:{1:0.0091, 2:0.0171, 3:0.0263, 4:0.0375, 5:0.0417}}
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

    # support = -0.037*real_k + 0.032*real_e + 0.034*(real_k*real_k) - 0.081*(real_k*real_e) + 0.030*(real_e*real_e) - 0.04
    # support = -0.040*real_k + 0.032*real_e - 0.054*real_d + 0.035*(real_k**2) - 0.083*(real_k*real_e) + \
    # 0.007*(real_k*real_d) + 0.029*(real_e**2) + 0.003*(real_e*real_d) + 0.032*(real_d**2) + 0.05
    print("The support is: ", support)
    
    #Measure time to compute maxent
    tic = time.time()

    # maxent_ur_up = unreg. unpert. maxent (vanilla)
    # maxent_ur_p = unreg. pert. maxent
    # maxent_r_up = reg. unpert. maxent 
    # maxent_r_p = reg. pert. maxent 

    #####################################################################
    cleaneddata_up = clean_preproc_data(directory)
    support_dict, two_wayc, three_wayc, four_wayc = marketbasket(cleaneddata_up, support)

    if len(two_wayc) + len(three_wayc) + len(four_wayc) > 100:
        feats = ExtractFeatures(cleaneddata_up.values, Mu=7)
    else:
        feats = ExtractFeatures(cleaneddata_up.values)
    
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
    
    opt_ur_up = Optimizer(feats)
    #Use LP to detect zero atoms 
    # opt_ur_up.exact_zero_detection(cleaneddata_up)

    soln_opt_ur_up = opt_ur_up.solver_optimize()
    if soln_opt_ur_up == None:
        print('Solution does not converge')
        return 

    maxent_ur_up, sum_prob_maxent_ur_up = compute_prob_exact(opt_ur_up, k, file_num, 0)

    print("Maxent Prob: (Unregularized, Unperturbed)", sum_prob_maxent_ur_up)
    print

    # 95% CI for width 
    opt_r_up = Optimizer_box(feats)
    #Use LP to detect zero atoms 
    # opt_r_up.exact_zero_detection(cleaneddata_up)
    
    soln_opt_r_up = opt_r_up.solver_optimize()
    if soln_opt_r_up == None:
        print('Solution does not converge')
        return 

    maxent_r_up, sum_prob_maxent_r_up = compute_prob_exact(opt_r_up, k, file_num, 0)

    print("Maxent Prob: (Regularized, Unperturbed)", sum_prob_maxent_r_up)
    print

    toc = time.time()
    print('Computational time for calculating maxent = {} seconds'.format(toc-tic))

    return (maxent_ur_up, maxent_r_up)

def calc_maxent_perturbed(file_num, directory, perturb_prob, k, width=None):
    """
    Calculate maximum entropy for each perturbation and width and return results
    """
    #tested for lambda = 2/3
    support_vals = {4:{1:0.0792, 2:0.063, 3:0.0671, 4:0.0799, 5:0.0671}, 
                    7:{1:0.0725, 2:0.0558, 3:0.0529, 4:0.0661, 5:0.0835},
                    10:{1:0.083, 2:0.0555, 3:0.0496, 4:0.0414, 5:0.0571},
                    15:{1:0.0091, 2:0.0171, 3:0.0263, 4:0.0375, 5:0.0417}}
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
    
    # support = -0.037*real_k + 0.032*real_e + 0.034*(real_k*real_k) - 0.081*(real_k*real_e) + 0.030*(real_e*real_e) - 0.04
    # support = -0.040*real_k + 0.032*real_e - 0.054*real_d + 0.035*(real_k**2) - 0.083*(real_k*real_e) + \
    # 0.007*(real_k*real_d) + 0.029*(real_e**2) + 0.003*(real_e*real_d) + 0.032*(real_d**2) + 0.05
    print("The support is: ", support)

    #Measure time to compute maxent
    tic = time.time()

    # maxent_ur_up = unreg. unpert. maxent (vanilla)
    # maxent_ur_p = unreg. pert. maxent
    # maxent_r_up = reg. unpert. maxent 
    # maxent_r_p = reg. pert. maxent 

    ######################################################################
    cleaneddata_p = clean_preproc_data_perturb(directory, perturb_prob)
    support_dict, two_wayc, three_wayc, four_wayc = marketbasket(cleaneddata_p, support)

    if len(two_wayc) + len(three_wayc) + len(four_wayc) > 100:
        feats = ExtractFeatures(cleaneddata_p.values, Mu=7)
    else:
        feats = ExtractFeatures(cleaneddata_p.values)
    
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
    
    opt_ur_p = Optimizer(feats)
    #Use LP to detect zero atoms 
    # opt_ur_p.exact_zero_detection(cleaneddata_p)

    soln_opt_ur_p = opt_ur_p.solver_optimize()
    if soln_opt_ur_p == None:
        print('Solution does not converge')
        return 

    maxent_ur_p, sum_prob_maxent_ur_p = compute_prob_exact(opt_ur_p, k, file_num, 1)

    print("Maxent Prob: (Unregularized, Perturbed)", sum_prob_maxent_ur_p)
    print

    # using 95% CI for width 
    opt_r_p = Optimizer_box(feats) 
    #Use LP to detect zero atoms 
    # opt_r_p.exact_zero_detection(cleaneddata_p)
    
    soln_opt_r_p = opt_r_p.solver_optimize()
    if soln_opt_r_p == None:
        print('Solution does not converge')
        return 

    maxent_r_p, sum_prob_maxent_r_p = compute_prob_exact(opt_r_p, k, file_num, 1)

    print("Maxent Prob: (Regularized, Perturbed)", sum_prob_maxent_r_p)
    print
    ############################################################################
    toc = time.time()
    print('Computational time for calculating maxent = {} seconds'.format(toc-tic))

    return (maxent_ur_p, maxent_r_p)


def main_up(file_num, k):
    '''
    Function to store multiple perturbation and width maximum entropy distributions in a single file
    '''
    # generating synthetic data 
    directory = '../dataset/d'+str(k)+'/synthetic_data_expt'+str(file_num)+'.csv'

    output = calc_maxent_unperturbed(file_num, directory, k=k)
    
    # for synthetic data 
    outfilename = '../output/d'+str(k)+'_expt2.3/syn_maxent_up'+str(file_num)+'.pickle'

    with open(outfilename, "wb") as outfile:
        pickle.dump(output, outfile)


def main_p(file_num, k):
    # generating synthetic data 
    directory = '../dataset/d'+str(k)+'/synthetic_data_expt'+str(file_num)+'.csv'

    perturb_prob = [0.01, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21]
    for p in perturb_prob:
        output = calc_maxent_perturbed(file_num, directory, perturb_prob=p, k=k)

        # for synthetic data 
        outfilename = '../output/d'+str(k)+'_expt2.3/syn_maxent_p'+str(file_num)+'_'+str(p)+'.pickle'

        with open(outfilename, "wb") as outfile:
            pickle.dump(output, outfile)

if __name__ == '__main__':
    # for synthetic data 
    num_dis = sys.argv[1]
    file_num = sys.argv[2]
    pert_flag = sys.argv[3]
    if pert_flag == str(1):
        main_p(file_num=int(file_num), k=int(num_dis))
    elif pert_flag == str(0):
        main_up(file_num=int(file_num), k=int(num_dis))
