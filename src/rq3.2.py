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
# from codebase.optimizer import Optimizer
# from codebase.optimizer_zeros import Optimizer
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

def get_mle_prob(cleaneddata):
    """
    Find the empirical probabilities of all 2**k disease combinations
    Args:
        cleaneddata: the pandas Dataframe of all patient vectors
    Returns: 
        emp: probability distribution vector for all combinations 
    """
    #ignore warnings for pandas dataframe handling 
    pd.options.mode.chained_assignment = None  # default='warn'

    data = cleaneddata
    size = data.shape[0]
    diseases = data.shape[1]
    cols = np.arange(diseases)
    data.columns = cols

    # initialize mle
    mle = []
    mle_sum = np.zeros(diseases+1)
    total_prob = 0.0

    ndata = data.groupby(list(data.columns)).size().to_frame('size').reset_index()
    ndata['size']/=size
    ndata['combi'] = ndata[list(data.columns)].values.tolist()
    ndata = ndata[['combi','size']]
   
    all_perms = itertools.product([0,1], repeat=diseases)
    for vec in all_perms:
        prob = ndata[ndata['combi'].apply(lambda x: x==list(vec))]['size'].values
        j = sum(vec)
        if prob.size==0:
            mle.append(0)
        else:
            mle.append(prob[0])
            mle_sum[j] += prob[0]
            total_prob += prob[0]

    # print('Total Probability, Emp:', total_prob)
    return mle, mle_sum


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
        # print('Vector:', vec, ' Probability: ', p_vec)
        j = sum(vec)
        maxent_sum_diseases[j] += p_vec
        total_prob += p_vec
        maxent_prob.append(p_vec) 
    
    print('Total Probability: ', total_prob)
    
    return maxent_prob, maxent_sum_diseases

def main(file_num=None, dataset_num=None, k=None, support=None):
    '''
    file_num: handler for reference to different lambda's
    k: num of diseases
    support: input different support values
    '''

    #tested for lambda = 2/3
    # support_vals = {4:{1:0.0792, 2:0.063, 3:0.0671, 4:0.0799, 5:0.0671}, 
    #                 7:{1:0.0725, 2:0.0558, 3:0.0529, 4:0.0661, 5:0.0835},
    #                 10:{1:0.083, 2:0.0555, 3:0.0496, 4:0.0414, 5:0.0571},
    #                 15:{1:0.0091, 2:0.0171, 3:0.0263, 4:0.0375, 5:0.0417}}
    # support = support_vals[k][file_num]

    exp = {1:0.8 , 2:1.2 , 3:1.6 , 4:2.0 , 5:2.4}
    size = {4: 50, 7: 125, 10: 250, 15:350}
    e = exp[file_num]
    d_size = size[k]
    real_k = (k-4)/(20-4)
    real_e = (e - 0.8)/(2.4-0.8)
    real_d = (d_size - 51)/(1000-51)
    
    # Random Forest Regression
    sup_model = read_model('../support_analysis/rfregr_model.pickle')
    covariates = np.array([real_k, real_e, real_d]).reshape(1,-1)
    support = sup_model.predict(covariates)[0]

    print("The support is: ", support)
    
    #Measure time to compute maxent
    tic = time.time()

    # generating synthetic data 
    if dataset_num == None:
        directory = '../dataset/d'+str(k)+'/synthetic_data_expt'+str(file_num)+'.csv'
    else:
        directory = '../dataset_s'+str(dataset_num)+'/d'+str(k)+'/synthetic_data_expt'+str(file_num)+'.csv'
    
    # for synthetic data 
    cleaneddata = clean_preproc_data(directory)
    # for real data 
    # cleaneddata = clean_preproc_data_real(directory)

    mle, mle_sum = get_mle_prob(cleaneddata)

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

    # print("\nThe constraints are:")
    # print('two_wayc', two_wayc)
    # print('three_wayc', three_wayc)
    # print('four_wayc', four_wayc)
    print('The total number of MBA constraints are: ', str(len(two_wayc) + len(three_wayc) + len(four_wayc)))
    print()

    # print(feats.feat_partitions)
    # width = 1
    opt = Optimizer_robust(feats, 1) 

    #Use LP to detect zero atoms 
    # opt.exact_zero_detection(cleaneddata)
    # opt.approximate_zero_detection(cleaneddata)
    
    soln_opt = opt.solver_optimize()
    if soln_opt == None:
        print('Solution does not converge')
        return 

    maxent, sum_prob_maxent = compute_prob_exact(opt)
    print()
    print('Empirical:', mle_sum)
    print("Maxent: " + str(sum_prob_maxent))
    if dataset_num == None:
        print("True distribution:" + str(read_prob_dist('../output/d'+str(k)+'/truedist_expt'+str(file_num)+'.pickle')))
    else:
        print("True distribution:" + str(read_prob_dist('../output_s'+str(dataset_num)+'/d'+str(k)+'/truedist_expt'+str(file_num)+'.pickle')))
   
    # for synthetic data 
    if dataset_num == None:
        outfilename = '../output/d'+str(k)+'_expt3.2_rob_ls/syn_emp_expt'+str(file_num)+'.pickle'
    else:        
        outfilename = '../output_s'+str(dataset_num)+'/d'+str(k)+'_expt3.2_rob_ls/syn_emp_expt'+str(file_num)+'.pickle'

    with open(outfilename, "wb") as outfile:
        pickle.dump((maxent, sum_prob_maxent, mle, mle_sum), outfile)
    
    toc = time.time()
    print('Computational time for calculating maxent = {} seconds'.format(toc-tic))
    print()

if __name__ == '__main__':
    # for synthetic data 
    num_dis = sys.argv[1]
    file_num = sys.argv[2]
    if len(sys.argv) > 3:
        dataset_num = sys.argv[3]
        main(file_num=int(file_num), dataset_num=int(dataset_num), k=int(num_dis))
    else:
        main(file_num=int(file_num), k=int(num_dis))


