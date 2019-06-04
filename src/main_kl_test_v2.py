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
from codebase.optimizer import Optimizer

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
    # print(sets_two, sets_three, sets_four)
    print(ltwo, lthree, lfour)

    """Output two way, three way and four way dictionaries as input for optimization"""
    for stwo in ltwo:
        two_way_dict[tuple(stwo)]=val_two
    for sthree in lthree:
        three_way_dict[tuple(sthree)]=val_three
    for sfour in lfour:
        four_way_dict[tuple(sfour)]=val_four
    return two_way_dict, three_way_dict, four_way_dict

def compute_prob_exact(optobj, prob_zeros):
    maxent_prob = []
    num_feats = optobj.feats_obj.data_arr.shape[1]
    maxent_diseases = np.zeros(num_feats+1)
    all_perms = itertools.product([0, 1], repeat=num_feats)
    all_perms = list(all_perms)
    print(all_perms[0])

    total_prob = 0.0    # finally should be very close to 1

    print("probability of zeros = ", prob_zeros)
    maxent_prob.append(prob_zeros)
    total_prob += prob_zeros
    maxent_diseases[0] = prob_zeros

    for tmp in all_perms[1:]:
    # for tmp in all_perms:
        vec = np.asarray(tmp)
        p_vec = optobj.prob_dist(vec)*(1-prob_zeros)
        j = sum(vec)
        maxent_diseases[j] += p_vec 
        total_prob += p_vec
        maxent_prob.append(p_vec) 


    print("Total_prob: " + str(total_prob))

    emp_prob = np.zeros(num_feats + 1)
    for vec in optobj.feats_obj.data_arr:
        j = sum(vec)
        emp_prob[j] += 1
    emp_prob /= optobj.feats_obj.data_arr.shape[0]
    
    print("maxent_diseases: " + str(maxent_diseases))
    return maxent_prob, maxent_diseases, emp_prob

def main(file_num=None):
    print("File num: " + str(file_num) + " has started")
    
    # support_data = {1:0.002, 2:0.002, 3:0.002, 4:0.002, 5:0.002, 6:0.005, \
    #     7:0.005, 8:0.005, 9:0.005, 10:0.005, 11:0.012, 12:0.012, 13:0.016, \
    #     14:0.014, 15:0.013, 16:0.02, 17:0.018, 18:0.021, 19:0.023, 20:0.025, 21:0.029, \
    #     22:0.026, 23:0.027, 24:0.029, 25:0.032}
    # support = support_data[file_num]

    support_data_overlap_nonzero = {1:0.002, 2:0.002, 3:0.002, 4:0.002, 5:0.002, 6:0.007, \
        7:0.008, 8:0.007, 9:0.007, 10:0.007, 11:0.012, 12:0.016, 13:0.019, \
        14:0.019, 15:0.018, 16:0.024, 17:0.028, 18:0.02, 19:0.028, 20:0.027, 21:0.04, \
        22:0.04, 23:0.041, 24:0.041, 25:0.047}
    support = support_data_overlap_nonzero[file_num]

 
    tic = time.time()

    # real data
    # directory = '../dataset/basket_sets.csv'
    # generating synthetic data 
    directory = '../dataset/d500_overlap/synthetic_data_expt'+str(file_num)+'.csv'
    
    cleaneddata=pd.read_csv(directory, error_bad_lines=False)
    
    cleaneddata = cleaneddata.loc[~(cleaneddata==0).all(axis=1)]
    non_zero_rows = cleaneddata.shape[0]
    prob_zeros = (500-non_zero_rows)/500

    print("The zero vector probabilities are: " + str(prob_zeros))

    two_wayc, three_wayc, four_wayc = marketbasket(cleaneddata, support)
    data_array = load_disease_data(directory)
    feats = ExtractFeatures(data_array)

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

    maxent, sum_prob_maxent, emp_prob = compute_prob_exact(opt, prob_zeros)
    print("writing to file")

    # for real data
    # outfilename = '../output/realdata_maxent.pickle'
    # for synthetic data 
    outfilename = '../output/d500_overlap_nonzero/syn_maxent_expt'+str(file_num)+'.pickle'

    with open(outfilename, "wb") as outfile:
        pickle.dump((maxent, sum_prob_maxent, emp_prob), outfile)

    toc = time.time()
    print('Computational time for calculating maxent = {} seconds'.format(toc-tic))

if __name__ == '__main__':
    file_num = sys.argv[1]
    main(int(file_num))

