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
from scipy.stats import power_divergence

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

# Get divergence of probability distributions
def divergence(p, q):
	return (p*np.log(p/q)).sum()

def calc_div(k):
	global df 
	div = []
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
			# kl_div = kl_divergence(p, q)
			pow_div, p_val = power_divergence(f_obs=q, f_exp=p, lambda_="cressie-read")
			# print('Power Divergence is: ', kl_div)
			# print('P value is: ', p_val)
		except FloatingPointError as e:
			print('Infinity')

		pow_div = round(pow_div, 4)
		div.append(pow_div)

		data_dict = {'Lambda':lambdas[ind],'# Diseases':k, 'Power Divergence':pow_div}
		df = df.append(data_dict, ignore_index=True)

	return div


def plot():
	global df
	divs = {}
	plt.style.use('seaborn-darkgrid')

	for ind, dis in enumerate(diseases):
		divs[dis] = calc_div(dis)
		plt.plot(lambdas, divs[dis], label=str(dis)+' diseases')

	# y_ticks = np.arange(0, 2, 0.4)
	# plt.yticks(y_ticks)
	plt.legend(fontsize=9)
	plt.title('Maximum Entropy for different lambda')
	plt.xlabel('Lambda (Exponential Distribution)')
	plt.ylabel('Power Divergence')
	plt.show()

	print('DataFrame is:\n', df)

	# df.to_csv('../output/expt1/d'+str(dis)+'.csv', index=False)

# globally accessible variables
diseases = [4,7,10,15]
lambdas = [0.42, 0.5, 0.63, 0.83, 1.25]
kls = {}
cols = ['Lambda', '# Diseases', 'Power Divergence']
df = pd.DataFrame(columns=cols)
plot()

