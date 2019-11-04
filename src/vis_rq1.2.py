'''
Python = 3.7 
matplotlib to plot figures
'''
#PYTHON3 
import numpy as np
import pickle 
import matplotlib.pyplot as plt
import csv
import pandas as pd
import sys
import os.path 
from collections import defaultdict

'''
TODO: Optimize readingo of the maxent file, it is happening twice
Plot all supports for all lambdas for a particular disease in a single plot 
'''

np.seterr(all='raise')

# Read the true prob. distribution for sum of diseases
def read_true_prob(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[0], prob[1]

# Read the maxent prob. distribution for sum of diseases
def read_maxent_prob(filename):
	with open(filename, "rb") as outfile:
		data = pickle.load(outfile,encoding='latin1')
	return data[0], data[1], data[2], data[3], data[4]

# Get kl divergence of probability distributions
def kl_divergence(p, q):
	return (p*np.log(p/q)).sum()

def calc_kl(i,k):
	constraints = []
	sup_metric = []
	kl = []
	global df 
	for j in range(12):
		true_file = '../output/d'+str(k)+'/truedist_expt'+str(i)+'.pickle'
		true_dist, true_prob = read_true_prob(true_file)

		maxent_file = '../output/d'+str(k)+'_expt1.1/syn_maxent_expt'+str(i)+'_s'+str(j)+'.pickle'
		maxent_dist, maxent_prob, emp_prob, num_constraints, support = read_maxent_prob(maxent_file)

		emp_prob = np.around(emp_prob, decimals=4)
		maxent_prob = np.around(maxent_prob, decimals=4)
		true_prob = np.around(true_prob, decimals=4)
		support = np.around(support, decimals=3)

		p = np.array(true_dist)
		q = np.array(maxent_dist)
		try:
			kl_div = kl_divergence(p, q)
		except FloatingPointError as e:
			print('Infinity')

		kl_div = round(kl_div, 4)
		constraints.append(num_constraints)
		sup_metric.append(support)
		kl.append(kl_div)

		data_dict = {'Lambda':lambdas[i-1], 'Support':support, 'Constraints':num_constraints, \
			 'KL Divergence':kl_div}
		df = df.append(data_dict, ignore_index=True)

	return constraints, sup_metric, kl


def plot(dis_num):
	global df
	plt.style.use('seaborn-darkgrid')

	for ind, l in enumerate(lambdas):
		cons[l], sups[l], kls[l] = calc_kl(ind+1, dis_num)
		plt.plot(sups[l], kls[l], label='Lambda :'+str(l))

	plt.legend(fontsize=9)
	plt.title('Maxent vs. Support: '+str(dis_num)+' diseases')
	plt.xlabel('Support')
	plt.ylabel('KL Divergence')
	plt.show()

	df.to_csv('../output/expt1.1/d'+str(dis_num)+'.csv', index=False)

# globally accessible variables
lambdas = [0.42, 0.5, 0.63, 0.83, 1.25]
cons = {}
sups = {}
kls = {}
cols = ['Lambda', 'Support', 'Constraints', 'KL Divergence']
df = pd.DataFrame(columns=cols)
dis_num = sys.argv[1]
plot(dis_num)

