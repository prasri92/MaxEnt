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

'''
Functions to read the probability distributions from stored files
'''
# Read the prob. distribution for sum of diseases
def read_exact_maxent(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[1]

# Read the empirical prob. distribution for sum of diseases
def read_emp_prob(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[2]

# Get kl divergence of probability distributions
def kl_divergence(p, q):
	return (p*np.log(p/q)).sum()

def compute_kl(support, num_feats):
	'''
	Compute the kl divergence among the exact maxent and approximate methods 
	'''
	file = '../output/real_subsets/top_'+str(num_feats)+'d_'+str(support)+'.pickle'

	emp_prob = read_emp_prob(file)
	maxent_prob = read_exact_maxent(file)

	p = (np.array(emp_prob))
	q = (np.array(maxent_prob))
	kl_div = kl_divergence(p,q)

	return emp_prob, maxent_prob, kl_div

def plot(num_feats):
	exact = []
	kl = []

	support = [0.0001, 0.0005, 0.001, 0.005, 0.01]

	for s in support:
		emp, ex, k = compute_kl(s, num_feats)
		exact.append(ex)
		kl.append(k)

	num_feats = num_feats
	xvec = [i for i in range(num_feats+1)]
	x_ticks = np.arange(0, num_feats+1, 1.0)
	plot_lims = [-1,  num_feats+1, -0.1, 1.0]

	emp = np.around(emp, decimals=4)

	for n in range(len(exact)):
		exact[n] = np.around(exact[n], decimals=4)
		kl[n] = np.around(kl[n], decimals=4)

	for ind, s in enumerate(support):
		plt.plot(xvec, exact[ind], label='Support: '+str(s))
	plt.plot(xvec, emp, 'k', label='Empirical')

	plt.axis(plot_lims)
	plt.xticks(x_ticks)
	plt.xlabel("Number of diseases per patient")
	plt.ylabel("Prob. of having x diseases")
	plt.title('Number of total diseases: '+str(num_feats))

	plt.legend(fontsize=6)
	row = []

	plt.subplots_adjust(top=0.85)
	plt.show()

if __name__ == '__main__':
	num_feats = sys.argv[1]
	plot(int(num_feats))