from __future__ import division
import pickle
import numpy as np
import itertools
from matplotlib import pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import sys


path_to_codebase = './codebase/'
sys.path.insert(0, path_to_codebase)
from codebase.utils import load_disease_data
from codebase.extract_features import ExtractFeatures
from codebase.optimizer import Optimizer

# filePath = '../data/Age50_DataExtract_fy.csv'
# filePath = '../toy_dataset/Age50_DataExtract_fy.csv'
filePath = '../dataset/basket_sets.csv'

def marketbasket(cleaneddata):
    frequent_itemsets = apriori(cleaneddata, min_support=0.004, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.001)
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
            tmp.append(int(j))
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
            tmp.append(int(j))
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
            tmp.append(int(j))
        lfour.append(tmp)
    print(sets_two, sets_three, sets_four)
    print(ltwo, lthree, lfour)

    """Output two way, three way and four way dictionaries as input for optimization"""
    for stwo in ltwo:
        two_way_dict[tuple(stwo)]=val_two
    for sthree in lthree:
        three_way_dict[tuple(sthree)]=val_three
    for sfour in lfour:
        four_way_dict[tuple(sfour)]=val_four
    return two_way_dict, three_way_dict, four_way_dict
    
# def compute_prob_exact(optobj):
#     maxent_prob = []
#     probsum = 0
#     numfeat = optobj.feats_obj.data_arr.shape[1]
#     for idx in range(2**numfeat):
#         b = format(idx, '0{}b'.format(numfeat))
#         r = [int(j) for j in b]
#         tmp = optobj.prob_dist(r)
#         probsum += tmp
#         maxent_prob.append(tmp)
#     print("Total probability", probsum)
#     return maxent_prob

def compute_prob_exact(optobj):
    maxent_prob = []
    num_feats = optobj.feats_obj.data_arr.shape[1]
    all_perms = itertools.product([0, 1], repeat=num_feats)
    total_prob = 0.0    # finally should be very close to 1
    for tmp in all_perms:
        vec = np.asarray(tmp)
        p_vec = optobj.prob_dist(vec)
        print(p_vec)
        total_prob += p_vec
        maxent_prob.append(p_vec) 
    return maxent_prob

def main():
    directory = '../dataset/basket_sets.csv'
    cleaneddata=pd.read_csv(directory, error_bad_lines=False)
    two_wayc, three_wayc, four_wayc = marketbasket(cleaneddata)
    data_array = load_disease_data(filePath)
    feats = ExtractFeatures(data_array)

    feats.set_two_way_constraints(two_wayc)
    feats.set_three_way_constraints(three_wayc)
    feats.set_four_way_constraints(four_wayc)

    feats.partition_features()
    print feats.feat_partitions

    opt = Optimizer(feats) 
    soln_opt = opt.solver_optimize()

    maxent = compute_prob_exact(opt)
    
    outfilename = '../output/top20diseases_actual.pickle'
    with open(outfilename, "wb") as outfile:
        pickle.dump(maxent, outfile)

if __name__ == '__main__':
    main()

'''

vals = [0, 1, 2, 3, 4, 5]
for v in vals:
    print str(v), compute_prob_exact(opt, v)

#### PLOTS #### 

#Calculate the max-ent probability of having x number of diseases
# total should add upto 1
num_feats = data_array.shape[1]
all_perms = itertools.product([0, 1], repeat=num_feats)
total_prob = 0.0    # finally should be very close to 1
mxt_prob = np.zeros(num_feats + 1)
for tmp in all_perms:
    vec = np.asarray(tmp)
    j = sum(vec)
    p_vec = opt.prob_dist(vec)
    total_prob += p_vec
    mxt_prob[j] += p_vec

emp_prob = np.zeros(num_feats + 1)
for vec in data_array:
    j = sum(vec)
    emp_prob[j] += 1
emp_prob /= data_array.shape[0] # N

print mxt_prob, emp_prob


# xvec = [i+1 for i in range(num_feats + 1)]
# # ~xvec
# x_ticks = np.arange(0, num_feats+2, 1.0)
# # ~x_ticks
# plot_lims = [0,  num_feats+2, -0.1, 1.0]
# # ~plot_lims
# # Both on same plot
# plt.figure()
# plt.plot(xvec, emp_prob, 'ro', label='empirical')  # empirical
# plt.plot(xvec, mxt_prob, 'bo', label='maxent')  # maxent
# plt.legend()
# plt.xticks(x_ticks)
# plt.axis(plot_lims)
# plt.show()
# # plt.savefig('../out/plot_merge_' + str(k_val) + '.png')
'''