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
from codebase.utils import load_disease_data
from codebase.extract_features import ExtractFeatures
from codebase.optimizer_old import Optimizer

def marketbasket(cleaneddata, support):
    frequent_itemsets = apriori(cleaneddata, min_support=support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.002)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequent_len"] = rules["consequents"].apply(lambda x: len(x))
    indices_two=list(rules.loc[(rules['antecedent_len'] == 1) & (rules['consequent_len'] == 1)].sort_values(by=['lift'], ascending=False).index.values)
    indices_three=list(rules.loc[((rules['antecedent_len'] == 2) & (rules['consequent_len'] == 1)) | ((rules['antecedent_len'] == 1) & (rules['consequent_len'] == 2)) ].sort_values(by=['lift'], ascending=False).index.values)
    indices_four=list(rules.loc[((rules['antecedent_len'] == 2) & (rules['consequent_len'] == 2)) | ((rules['antecedent_len'] == 1) & (rules['consequent_len'] == 3)) | ((rules['antecedent_len'] == 3) & (rules['consequent_len'] == 1)) ].sort_values(by=['lift'], ascending=False).index.values)
    """Declare empty sets to store itemsets of sizes two, three and four"""
    sets_two = set()
    sets_three = set()
    sets_four = set()
    val_two=(1,1)
    val_three=(1,1,1)
    val_four=(1,1,1,1)
    two_way_dict={}
    three_way_dict={}
    four_way_dict={}
    print("Loading the data into a dictionary")
    """Add frozen sets of pairs, triplets and quadruplets to declared empty sets"""
    for itwo in indices_two:
        ltwo = []
        a = rules.iloc[itwo]['antecedents']
        b = rules.iloc[itwo]['consequents']
        sets_two.add(a.union(b))
    ltwo=[]
    for i in sets_two:
        tmp = []
        for j in i:
            tmp.append(int(float(j)))
        ltwo.append(tmp)

    for ithree in indices_three:
        lthree = []
        a = rules.iloc[ithree]['antecedents']
        b = rules.iloc[ithree]['consequents']
        sets_three.add(a.union(b))
    lthree=[]
    for i in sets_three:
        tmp = []
        for j in i:
            tmp.append(int(float(j)))
        lthree.append(tmp)

    for ifour in indices_four:
        lfour = []
        a = rules.iloc[ifour]['antecedents']
        b = rules.iloc[ifour]['consequents']
        sets_four.add(a.union(b))
    lfour = []
    for i in sets_four:
        tmp = []
        for j in i:
            tmp.append(int(float(j)))
        lfour.append(tmp)
    print(ltwo, lthree, lfour)

    """Output two way, three way and four way dictionaries as input for optimization"""
    for stwo in ltwo:
        two_way_dict[tuple(stwo)]=val_two
    for sthree in lthree:
        three_way_dict[tuple(sthree)]=val_three
    for sfour in lfour:
        four_way_dict[tuple(sfour)]=val_four
    return two_way_dict, three_way_dict, four_way_dict

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
    print("File num: " + str(file_num) + " has started")
    
    # support_data_overlap_nonzero = {1:0.003, 2:0.002, 3:0.002, 4:0.001, 5:0.002, 6:0.007, \
    #     7:0.008, 8:0.01, 9:0.006, 10:0.007, 11:0.016, 12:0.015, 13:0.016, \
    #     14:0.014, 15:0.014, 16:0.028, 17:0.028, 18:0.02, 19:0.02, 20:0.02, 21:0.035, \
    #     22:0.04, 23:0.041, 24:0.04, 25:0.035}  
    # support = support_data_overlap_nonzero[file_num]

    # for four diseases
    support = 0.01
 
    tic = time.time()

    # real data
    # directory = '../dataset/basket_sets.csv'
    # generating synthetic data 
    directory = '../dataset/d'+str(size)+'_4/synthetic_data_expt'+str(file_num)+'.csv'

    cleaneddata=pd.read_csv(directory, error_bad_lines=False)
    cleaneddata = cleaneddata.loc[~(cleaneddata==0).all(axis=1)]
    non_zero_rows = cleaneddata.shape[0]
    prob_zeros = (int(size)-non_zero_rows)/int(size)

    two_wayc, three_wayc, four_wayc = marketbasket(cleaneddata, support)
    # two_wayc = {}
    # two_wayc = {(1, 3): (1, 1)} 

    # two_wayc = {(1, 0): (1, 1), (0, 3): (1, 1)} 
    # two_wayc = {(1, 0): (1, 1), (0, 3): (1, 1) , (3, 2): (1, 1), (0, 2): (1, 1)}
    # two_wayc = {(1, 0): (1, 1), (0, 3): (1, 1), (3, 2): (1, 1), (0, 2): (1, 1), (1, 3): (1, 1), (1, 2): (1, 1)}

    three_wayc = {}
    # three_wayc = {(1, 0, 3): (1, 1, 1), (0, 3, 2): (1, 1, 1)}  
    # three_wayc = {(1, 0, 3): (1, 1, 1), (0, 3, 2): (1, 1, 1), (1, 3, 2): (1, 1, 1), (1, 0, 2): (1, 1, 1)}
    
    four_wayc = {}
    # four_wayc = {(1, 0, 3, 2): (1, 1, 1, 1)}
    
    feats = ExtractFeatures(cleaneddata.values)

    feats.set_two_way_constraints(two_wayc)
    feats.set_three_way_constraints(three_wayc)
    feats.set_four_way_constraints(four_wayc)

    feats.partition_features()
    print(feats.feat_partitions)

    print('two_wayc', two_wayc)
    print('three_wayc', three_wayc)
    print('four_wayc', four_wayc)

    opt = Optimizer(feats) 
    soln_opt = opt.solver_optimize()
    print("Optimizer is done. Computing probabilities")

    maxent, sum_prob_maxent, emp_prob = compute_prob_exact(opt, prob_zeros, size)
    print("Empirical: " +str(emp_prob))
    print("Maxent: " + str(sum_prob_maxent))
    print("True distribution:" + str(read_prob_dist('../output/d'+str(size)+'_4/truedist_expt'+str(file_num)+'.pickle')))
    
    print("writing to file")

    # for real data'support_'+str(support)+
    # outfilename = '../output/realdata_maxent.pickle'
    # for synthetic data 
    outfilename = '../output/d'+str(size)+'_4_nonzeros/syn_maxent_expt'+str(file_num)+'.pickle'#'_constraints_'+str(trial)+'.pickle' #

    with open(outfilename, "wb") as outfile:
        pickle.dump((maxent, sum_prob_maxent, emp_prob), outfile)

    toc = time.time()
    print('Computational time for calculating maxent = {} seconds'.format(toc-tic))

if __name__ == '__main__':
    file_num = sys.argv[1]
    size = sys.argv[2]
    # trial = sys.argv[3]
    main(file_num=int(file_num), size=size)#, trial=trial)

