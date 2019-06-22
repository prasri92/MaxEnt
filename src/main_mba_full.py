from __future__ import division
import pickle
import numpy as np
import itertools
#from matplotlib import pyplot as plt

import sys
#path_to_codebase = '/mnt/Study/umass/sem3/maxEnt/src/codebase/'
path_to_codebase = './codebase/'
sys.path.insert(0, path_to_codebase)
from codebase.utils import load_disease_data
from codebase.extract_features import ExtractFeatures
from codebase.optimizer import Optimizer

# filePath = '../data/test1-fy.csv'
filePath = '../data/2010-2014-fy.csv'
entropy_est = 'JAMES-STEIN'
k_val = 40 

# Creating the 2-way constraint dict from MBA analysis
two_wayc = {}
keys = [(8,13), (4,13), (4,8), (13, 30), (13,15), (8,30), (36,37), (30,31), 
        (13, 31), (4,30), (13, 29), (8,15), (13,32), (13,35), (30,32), (13,37), 
        (13,23), (31,32), (8,31), (4, 31), (4,15)]

common_val = (1,1)  # Since MBA only looks for (1,1) pairs
for k in keys:
    two_wayc[k] = common_val

# Creating the 3-way constraint dict from MBA analysis
three_wayc = {}
keys = [(4,8,13), (8,13,30), (4,13,30), (4,13,15), (4,8,30), (8,13,29), 
        (8,13,31), (8,13,32), (4,8,15), (8,13,37), (8,13,35), (13,30,31), 
        (13,15,14), (8,13,14), (8,13,23), (4,13,29), (3,8,13), (8,13,28),
        (4,13,35)]

common_val = (1,1,1)  # Since MBA only looks for (1,1,1) pairs
for k in keys:
    three_wayc[k] = common_val

# Creating the 4-way constraint dict from MBA analysis
four_wayc = {}
keys = [(4,8,13,30), (4,8,13,15), (8,13,14,15), (4,8,13,31), (4,8,13,29), 
        (8,13,15,30), (4,8,13,23), (4,8,13,35), (4,8,13,32), (8,13,30,31), 
        (4,8,13,15), (4,13,14,15), (8,13,30,32), (8,13,15,23), (4,8,13,37), 
        (4,8,13,11), (4,8,13,27), (8,13,31,32), (8,13,30,35), (4,8,13,36) ]

common_val = (1,1,1,1)  # Since MBA only looks for (1,1,1) pairs
for k in keys:
    four_wayc[k] = common_val

data_array = load_disease_data(filePath)

feats = ExtractFeatures(data_array, entropy_est, k_val)

feats.set_two_way_constraints(two_wayc)
feats.set_three_way_constraints(three_wayc)
feats.set_four_way_constraints(four_wayc)
feats.partition_features()

print("Starting the optimizer")
opt = Optimizer(feats) 
soln_opt = opt.solver_optimize()
print("Optimization over")

outfilename = '../out/pickles/obj_full_mba.pk'
with open(outfilename, "wb") as outfile:
    pickle.dump(opt, outfile)
