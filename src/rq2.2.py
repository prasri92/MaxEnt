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

def calc_maxent_unperturbed(file_num, directory, k, width=None):
    """
    Calculate maximum entropy for each perturbation and width and return results
    """
    support_vals = {4:{1:0.045, 2:0.063, 3:0.063, 4:0.081, 5:0.082}, 
                    7:{1:0.018, 2:0.036, 3:0.063, 4:0.063, 5:0.1},
                    10:{1:0.009, 2:0.018, 3:0.027, 4:0.036, 5:0.054},
                    15:{1:0.003, 2:0.005, 3:0.011, 4:0.020, 5:0.027}}
    support = support_vals[k][file_num]
    
    # exp = {1:0.8 , 2:1.2 , 3:1.6 , 4:2.0 , 5:2.4}
    # size = {4: 50, 7: 125, 10: 250, 15:350}
    # e = exp[file_num]
    # d_size = size[k]
    # real_k = (k-4)/(20-4)
    # real_e = (e - 0.8)/(2.4-0.8)
    # real_d = (d_size - 51)/(1000-51)
    # # support = -0.037*real_k + 0.032*real_e + 0.034*(real_k*real_k) - 0.081*(real_k*real_e) + 0.030*(real_e*real_e) - 0.04
    # support = -0.040*real_k + 0.032*real_e - 0.054*real_d + 0.035*(real_k**2) - 0.083*(real_k*real_e) + \
    # 0.007*(real_k*real_d) + 0.029*(real_e**2) + 0.003*(real_e*real_d) + 0.032*(real_d**2) + 0.05
    # print("The support is: ", support)
    
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

    #Use LP to detect zero atoms 
    # opt.exact_zero_detection(cleaneddata)
    # opt.approximate_zero_detection(cleaneddata)
    
    opt_ur_up = Optimizer(feats)
    soln_opt_ur_up = opt_ur_up.solver_optimize()
    if soln_opt_ur_up == None:
        print('Solution does not converge')
        return 

    maxent_ur_up, sum_prob_maxent_ur_up = compute_prob_exact(opt_ur_up)

    print("Maxent Prob: (Unregularized, Unperturbed)", sum_prob_maxent_ur_up)
    print

    # Use regularization methods of exponential prior 
    # lambdas = {1:0.42, 2:0.5, 3:0.62, 4:0.83, 5:1.25}
    # opt_r_up = Optimizer_exp(feats, lambdas[file_num])

    # Use regularization methods of box constraints
    # width = {4:50, 7:225, 10:350, 15:350}
    # opt_r_up = Optimizer_box(feats, width[k]) 

    # for testing different widths 
    opt_r_up = Optimizer_box(feats, width)
    
    soln_opt_r_up = opt_r_up.solver_optimize()
    if soln_opt_r_up == None:
        print('Solution does not converge')
        return 

    maxent_r_up, sum_prob_maxent_r_up = compute_prob_exact(opt_r_up)

    print("Maxent Prob: (Regularized, Unperturbed)", sum_prob_maxent_r_up)
    print

    toc = time.time()
    print('Computational time for calculating maxent = {} seconds'.format(toc-tic))

    return (maxent_ur_up, maxent_r_up)

def calc_maxent_perturbed(file_num, directory, perturb_prob, k, width=None):
    """
    Calculate maximum entropy for each perturbation and width and return results
    """
    support_vals = {4:{1:0.045, 2:0.063, 3:0.063, 4:0.081, 5:0.082}, 
                    7:{1:0.018, 2:0.036, 3:0.063, 4:0.063, 5:0.1},
                    10:{1:0.009, 2:0.018, 3:0.027, 4:0.036, 5:0.054},
                    15:{1:0.003, 2:0.005, 3:0.011, 4:0.020, 5:0.027}}
    support = support_vals[k][file_num]

    # exp = {1:0.8 , 2:1.2 , 3:1.6 , 4:2.0 , 5:2.4}
    # size = {4: 50, 7: 125, 10: 250, 15:350}
    # e = exp[file_num]
    # d_size = size[k]
    # real_k = (k-4)/(20-4)
    # real_e = (e - 0.8)/(2.4-0.8)
    # real_d = (d_size - 51)/(1000-51)
    # # support = -0.037*real_k + 0.032*real_e + 0.034*(real_k*real_k) - 0.081*(real_k*real_e) + 0.030*(real_e*real_e) - 0.04
    # support = -0.040*real_k + 0.032*real_e - 0.054*real_d + 0.035*(real_k**2) - 0.083*(real_k*real_e) + \
    # 0.007*(real_k*real_d) + 0.029*(real_e**2) + 0.003*(real_e*real_d) + 0.032*(real_d**2) + 0.05
    # print("The support is: ", support)
    
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

    #Use LP to detect zero atoms 
    # opt.exact_zero_detection(cleaneddata)
    # opt.approximate_zero_detection(cleaneddata)
    
    opt_ur_p = Optimizer(feats)
    soln_opt_ur_p = opt_ur_p.solver_optimize()
    if soln_opt_ur_p == None:
        print('Solution does not converge')
        return 

    maxent_ur_p, sum_prob_maxent_ur_p = compute_prob_exact(opt_ur_p)

    print("Maxent Prob: (Unregularized, Perturbed)", sum_prob_maxent_ur_p)
    print

    # Use regularization methods of exponential prior 
    # lambdas = {1:0.42, 2:0.5, 3:0.62, 4:0.83, 5:1.25}
    # opt_r_p = Optimizer_exp(feats, lambdas[file_num])

    # Use regularization methods of box constraints
    # width = {4:50, 7:225, 10:350, 15:350}
    # opt_r_p = Optimizer_box(feats, width[k]) 

    # for testing different widths 
    opt_r_p = Optimizer_box(feats, width)
    
    soln_opt_r_p = opt_r_p.solver_optimize()
    if soln_opt_r_p == None:
        print('Solution does not converge')
        return 

    maxent_r_p, sum_prob_maxent_r_p = compute_prob_exact(opt_r_p)

    print("Maxent Prob: (Regularized, Perturbed)", sum_prob_maxent_r_p)
    print
    ############################################################################
    toc = time.time()
    print('Computational time for calculating maxent = {} seconds'.format(toc-tic))

    return (maxent_ur_p, maxent_r_p)


def main_up(file_num, k, dataset=None):
    '''
    Function to store multiple perturbation and width maximum entropy distributions in a single file
    '''
    # generating synthetic data 
    if dataset==None:
        directory = '../dataset/d'+str(k)+'/synthetic_data_expt'+str(file_num)+'.csv'
    else:
        directory = '../dataset_s'+str(dataset)+'/d'+str(k)+'/synthetic_data_expt'+str(file_num)+'.csv'
    '''
    if k == 15:
        # widths = [70, 210, 350, 490, 630]
        # widths = [0.0001, 0.001, 0.01, 1, 350, 3500]
        # widths = [0.01, 1, 350, 600, 800, 950]
        widths = [1, 70, 175, 350]
    elif k == 10:
        # widths = [50, 150, 250, 350, 450]
        # widths = [0.0001, 0.001, 0.01, 1, 250, 2500]
        # widths = [1, 50, 250, 500, 5000, 10000000]
        # widths = [0.01, 1, 250, 450, 600, 800]   
        widths = [1, 50, 125, 250]
    elif k == 7:
        # widths = [25, 75, 125, 175, 225]
        # widths = [0.0001, 0.001, 0.01, 1, 125, 1250]
        # widths = [1, 50, 250, 500, 5000, 10000000]
        # widths = [0.01, 1, 125, 300, 450, 600]
        widths = [1, 25, 62.5, 125]
    elif k == 4:
        # widths = [10,30,50,70,90]
        # widths = [0.0001, 0.001, 0.01, 1, 50, 500]
        # widths = [1, 50, 250, 500, 5000, 10000000]
        # widths = [0.01, 1, 50, 120, 250, 400]
        widths = [1, 10, 25, 50]
    '''
    widths = [0.002, 0.2, 0.5, 1]
    # file_width = file_width = [1,2,3,4,5,6]
    file_width = [1,2,3,4]
    # IF testing for different widths (also change file name)
    for ind, w in enumerate(widths):
        output = calc_maxent_unperturbed(file_num, directory, k=k, width=w)
    
        # for synthetic data 
        if dataset==None:
            outfilename = '../output/d'+str(k)+'_expt2.2/syn_maxent_up'+str(file_num)+'_w'+str(file_width[ind])+'.pickle'
        else:
            outfilename = '../output_s'+str(dataset)+'/d'+str(k)+'_expt2.2/syn_maxent_up'+str(file_num)+'_w'+str(file_width[ind])+'.pickle'

        with open(outfilename, "wb") as outfile:
            pickle.dump(output, outfile)


def main_p(file_num, k, dataset=None):
    # generating synthetic data 
    if dataset==None:
        directory = '../dataset/d'+str(k)+'/synthetic_data_expt'+str(file_num)+'.csv'
    else:
        directory = '../dataset_s'+str(dataset)+'/d'+str(k)+'/synthetic_data_expt'+str(file_num)+'.csv'
    '''
    if k == 15:
        # widths = [70, 210, 350, 490, 630]
        # widths = [0.0001, 0.001, 0.01, 1, 350, 3500]
        # widths = [0.01, 1, 350, 600, 800, 950]
        widths = [1, 70, 175, 350]
    elif k == 10:
        # widths = [50, 150, 250, 350, 450]
        # widths = [0.0001, 0.001, 0.01, 1, 250, 2500]
        # widths = [1, 50, 250, 500, 5000, 10000000]  
        # widths = [0.01, 1, 250, 450, 600, 800]    
        widths = [1, 50, 125, 250]
    elif k == 7:
        # widths = [25, 75, 125, 175, 225]
        # widths = [0.0001, 0.001, 0.01, 1, 125, 1250]
        # widths = [1, 50, 250, 500, 5000, 10000000]        
        # widths = [0.01, 1, 125, 300, 450, 600]
        widths = [1, 25, 62.5, 125]
    elif k == 4:
        # widths = [10,30,50,70,90]
        # widths = [0.0001, 0.001, 0.01, 1, 50, 500]
        # widths = [1, 50, 250, 500, 5000, 10000000]
        # widths = [0.01, 1, 50, 120, 250, 400]
        widths = [1, 10, 25, 50]
    '''
    widths = [0.002, 0.2, 0.5, 1]
    # file_width = [1,2,3,4,5,6]
    file_width = [1,2,3,4]
    # IF testing for different widths (also change file name)
    for ind,w in enumerate(widths):
        perturb_prob = [0.01, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21]
        for p in perturb_prob:
            output = calc_maxent_perturbed(file_num, directory, perturb_prob=p, k=k, width=w)

            # for synthetic data 
            if dataset==None:
                outfilename = '../output/d'+str(k)+'_expt2.2/syn_maxent_p'+str(file_num)+'_p'+str(p)+'_w'+str(file_width[ind])+'.pickle'
            else:
                outfilename = '../output_s'+str(dataset)+'/d'+str(k)+'_expt2.2/syn_maxent_p'+str(file_num)+'_p'+str(p)+'_w'+str(file_width[ind])+'.pickle'
            
            with open(outfilename, "wb") as outfile:
                pickle.dump(output, outfile)

if __name__ == '__main__':
    # for synthetic data 
    num_dis = sys.argv[1]
    file_num = sys.argv[2]
    pert_flag = sys.argv[3]
    if len(sys.argv) <= 4:
        if pert_flag == str(0):
            main_p(file_num=int(file_num), k=int(num_dis))
        elif pert_flag == str(1):
            main_up(file_num=int(file_num), k=int(num_dis))
    else:
        dataset = sys.argv[4]
        if pert_flag == str(0):
            main_p(file_num=int(file_num), k=int(num_dis), dataset=int(dataset))
        elif pert_flag == str(1):
            main_up(file_num=int(file_num), k=int(num_dis), dataset=int(dataset))