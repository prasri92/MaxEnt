from __future__ import division
import pickle
import numpy as np
import itertools
from matplotlib import pyplot as plt

import sys

#TODO: change to relative paths to maxent repository 
path_to_codebase = '/mnt/Study/umass/sem3/maxEnt/src/codebase/'
# path_to_codebase = './codebase/'
sys.path.insert(0, path_to_codebase)
from codebase.utils import load_disease_data
from codebase.extract_features import ExtractFeatures
# from codebase.optimizer import Optimizer
from codebase.optimizer import Optimizer

# filePath = '../data/Age50_DataExtract_fy.csv'
filePath = '../toy_dataset/Age50_DataExtract_fy.csv'

entropy_estimator = 'JAMES-STEIN'
k_val = 7

# Creating the 2-way constraint dict from MBA analysis
two_wayc = {}
keys = [(0,1), (3,6), (3,4), (1,4)]


common_val = (1,1)  # Since MBA only looks for (1,1) pairs
for k in keys:
    two_wayc[k] = common_val


# Creating the 3-way constraint dict from MBA analysis
three_wayc = {}
keys = [(0,1,4), (3,6,4)]  


common_val = (1,1,1)  # Since MBA only looks for (1,1,1) pairs
for k in keys:
    three_wayc[k] = common_val


# Creating the 4-way constraint dict from MBA analysis
four_wayc = {}
keys = [(0,1,3,4)]

common_val = (1,1,1,1)  # Since MBA only looks for (1,1,1,1) pairs 
for k in keys:
    four_wayc[k] = common_val

data_array = load_disease_data(filePath)
feats = ExtractFeatures(data_array, entropy_estimator, k_val)

feats.set_two_way_constraints(two_wayc)
feats.set_three_way_constraints(three_wayc)
feats.set_four_way_constraints(four_wayc)

# feats.compute_topK_feats()

feats.partition_features()
print feats.feat_partitions

opt = Optimizer(feats) 
soln_opt = opt.solver_optimize()

# opt.compare_marginals()

# m2, e2 = opt.compare_constraints_2way()
# print m2
# print e2

# m3, e3 = opt.compare_constraints_3way()
# print m3
# print e3

# m4, e4 = opt.compare_constraints_4way()
# print m4
# print e4



def iter_nCk_binvect(n, k):
    for bits in itertools.combinations(range(n), k):
        tmp = np.zeros(n)
#         print bits
        tmp[list(bits)] = 1
        yield tmp


def compute_prob_exact(optobj, k):
    probsum = 0
    maxi, v = 0, None
    numfeat = optobj.feats_obj.data_arr.shape[1]
    for vec in iter_nCk_binvect(numfeat, k):
        tmp = optobj.prob_dist(vec)
        probsum += tmp
        if tmp >= maxi:
            maxi = tmp
            v = vec
    print(maxi, v)
    return probsum


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