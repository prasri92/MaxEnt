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

#work on it later LOTS TO BE DONE 

np.seterr(all='raise')
lambdas = [0.42, 0.5, 0.62, 0.83, 1.25]

# Read the true prob. distribution for sum of diseases
def read_true_prob(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[0], prob[1]

# Read the maxent prob. distribution for sum of diseases
def read_maxent_prob(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[0], prob[1], prob[2]

# Get kl divergence of probability distributions
def kl_divergence(p, q):
	return (p*np.log(p/q)).sum()

def calc_kl(i,k,j):
	kls = []
	cols = ['Lambdas','KL Div','# of diseases','Empirical','MaxEnt','True']
	df = pd.Dataframe(colums=cols)

	true_file = '../output/d'+str(k)+'/truedist_expt'+str(i)+'.pickle'
	true_dist, true_prob = read_true_prob(true_file)

	maxent_file = '../output/d'+str(k)+'/syn_maxent_expt'+str(i)+'.pickle'
	maxent_dist, maxent_prob, emp_prob = read_maxent_prob(maxent_file)

	emp_prob = np.around(emp_prob, decimals=4)
	maxent_prob = np.around(maxent_prob, decimals=4)
	true_prob = np.around(true_prob, decimals=4)

	p = np.array(true_dist)
	q = np.array(maxent_dist)
	try:
		kl_div = kl_divergence(p, q)
	except FloatingPointError as e:
		print('Infinity')

	kl_div = round(kl_div, 4)
	kls.append(kl_div)

	return kls

def plot(file_num, dis_num, supports):
	diseases = [4,7,10,15]
	all_vals = {}

	for dis in diseases:
		all_vals[d] = calc_kl()

	for j,s in enumerate(supports):
		kls.append(calc_kl(file_num, dis_num, j=j))
		print(np.around(s, 3), ' : ', kls[j])

	plt.style.use('seaborn-darkgrid')
	plt.plot(supports, kls, marker='o')
	plt.legend(fontsize=9)
	plt.title('Maximum Entropy vs. Support')
	plt.xlabel('Support')
	plt.ylabel('KL Divergence')
	plt.show()

plot()