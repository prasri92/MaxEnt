from __future__ import division
import pickle
import numpy as np
import itertools

import matplotlib
matplotlib.use('agg')  # fix Qt errors for display when using on serverj
from matplotlib import pyplot as plt

import sys
#path_to_codebase = '/mnt/Study/umass/sem3/maxEnt/src/codebase/'
path_to_codebase = './codebase/'
sys.path.insert(0, path_to_codebase)
from codebase.utils import load_disease_data
from codebase.extract_features import ExtractFeatures
from codebase.optimizer import Optimizer

filePath = '../data/test1-fy.csv'
entropy_est = 'JAMES-STEIN'
k_val = 20
data_array = load_disease_data(filePath)

#feats = ExtractFeatures(data_array, entropy_est, k_val)
#feats.compute_topK_feats_approx()
feat_file = "../out/pickles/feats_obj_red.pk"
with open(feat_file, "rb") as rfile:
    feats = pickle.load(rfile)

# sanity check
assert(k_val == feats.K)

#opt = Optimizer(feats)
#soln_opt = opt.solver_optimize()
opt_file = "../out/pickles/opt_obj_red.pk"
with open(opt_file, "rb") as ofile:
    opt = pickle.load(ofile)
    solun_opt = pickle.load(ofile)
    #optlist = pickle.load(ofile)


###############################
# DANGER CODE BELOW
# DONOT RUN ON PERSONAL PC
# WILL EAT UP ALL THE RAM!
################################

m1, m2 = opt.compare_marginals()
c1, c2 = opt.compare_constraints()
print m1, m2
print c1, c2

#### PLOTS for comparisons #### 
num_feats = data_array.shape[1]

all_perms = itertools.product([0, 1], repeat=num_feats)
mxt_prob = np.zeros(num_feats)

for tvec in all_perms:
    vec = np.asarray(tvec)
    for j in range(num_feats):
        if sum(vec) == j:
            mxt_prob[j] += opt.prob_dist(vec)            
            break


emp_prob = np.zeros(num_feats)
for vec in data_array:
    for j in range(num_feats):
        if sum(vec) == j:
            emp_prob[j] += 1
            break

emp_prob = emp_prob/data_array.shape[0]


xvec = [i+1 for i in range(num_feats)]
x_ticks = np.arange(0, num_feats+2, 1.0)
plot_lims = [0,  num_feats+2, -0.1, 1.0]

# Both on same plot
plt.figure()
plt.plot(xvec, emp_prob, 'ro')  # empirical
plt.plot(xvec, mxt_prob, 'bo')  # maxent
plt.xticks(x_ticks)
plt.axis(plot_lims)
plt.savefig('../out/newplots/plot-reduced-fy-' + str(k_val) + '.png')


# # Difference Plot
xvec = [i+1 for i in range(num_feats)]
x_ticks = np.arange(0, num_feats+2, 1.0)
plot_lims = [0,  num_feats+2, -0.5, 0.5]

diff_vec = emp_prob - mxt_prob
plt.figure()
plt.plot(xvec, diff_vec, 'go')
plt.xticks(x_ticks)
plt.axis(plot_lims)
plt.savefig('../out/newplots/plot-reduced-fy-diff' + str(k_val) + '.png')
