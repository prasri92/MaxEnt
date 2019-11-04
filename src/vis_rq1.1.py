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

# Read the true prob. distribution for sum of diseases
def read_true_prob(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[0], prob[1]

# Read the maxent prob. distribution for sum of diseases
def read_maxent_prob(filename):
	with open(filename, "rb") as outfile:
		data = pickle.load(outfile,encoding='latin1')
	return data[0], data[1], data[2]

# Get kl divergence of probability distributions
def kl_divergence(p, q):
	return (p*np.log(p/q)).sum()

def calc_kl(k):
	global df 
	kl = []
	for ind, l in enumerate(lambdas):
		true_file = '../output/d'+str(k)+'/truedist_expt'+str(ind+1)+'.pickle'
		true_dist, true_prob = read_true_prob(true_file)

		maxent_file = '../output/d'+str(k)+'/syn_maxent_expt'+str(ind+1)+'.pickle'
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
		kl.append(kl_div)

		data_dict = {'Lambda':lambdas[ind],'# Diseases':k, 'KL Divergence':kl_div}
		df = df.append(data_dict, ignore_index=True)

	return kl


def plot():
	global df
	kls = {}
	plt.style.use('seaborn-darkgrid')

	for ind, dis in enumerate(diseases):
		kls[dis] = calc_kl(dis)
		plt.plot(lambdas, kls[dis], label=str(dis)+' diseases')

	y_ticks = np.arange(0, 2, 0.4)
	plt.yticks(y_ticks)
	plt.legend(fontsize=9)
	plt.title('Maxent vs. Lambda')
	plt.xlabel('Lambda (Exponential Distribution')
	plt.ylabel('KL Divergence')
	plt.show()

	df.to_csv('../output/expt1/d'+str(dis)+'.csv', index=False)

# globally accessible variables
diseases = [4,7,10,15]
lambdas = [0.42, 0.5, 0.63, 0.83, 1.25]
kls = {}
cols = ['Lambda', '# Diseases', 'KL Divergence']
df = pd.DataFrame(columns=cols)
plot()

