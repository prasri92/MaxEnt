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
from scipy.spatial import distance
import itertools

'''
Visualize the differences in Jensen Shannon divergence values vs. Power divergence 
for different values of lambda, provide justification for the same
'''

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
	j_div = []
	p_div1 = []
	p_div2 = []
	p_div3 = []
	p_div4 = []
	p_div5 = []
	for ind, l in enumerate(lambdas):
		true_file = '../output/d'+str(k)+'/truedist_expt'+str(ind+1)+'.pickle'
		true_dist, true_prob = read_true_prob(true_file)

		maxent_file = '../output/d'+str(k)+'/syn_maxent_expt'+str(ind+1)+'.pickle'
		maxent_dist, maxent_prob, emp_prob = read_maxent_prob(maxent_file)

		emp_prob = np.around(emp_prob, decimals=4)
		maxent_prob = np.around(maxent_prob, decimals=4)
		true_prob = np.around(true_prob, decimals=4)
		
		# t = true_dist
		# m = maxent_dist
		# t = np.around(t, 4)
		# m = np.round(m, 4)

		# print('Exponent is: ', l)
		# all_perms = itertools.product([0,1],repeat=4)
		# i=0
		# for vec in all_perms:
		# 	print('Vector: ', vec, ' T: ', t[i], ' M: ', m[i])
		# 	i+=1
	
		# np.seterr(all='ignore')
		p = np.array(true_dist)
		q = np.array(maxent_dist)
		try:
			# kl_div = kl_divergence(p, q)
			js_div = distance.jensenshannon(p, q)
			pow_div1, p_val = power_divergence(f_obs=q, f_exp=p, lambda_="cressie-read")
			pow_div2, p_val = power_divergence(f_obs=q, f_exp=p, lambda_="neyman")
			pow_div3, p_val = power_divergence(f_obs=q, f_exp=p, lambda_="freeman-tukey")
			pow_div4, p_val = power_divergence(f_obs=q, f_exp=p, lambda_="mod-log-likelihood")
			pow_div5, p_val = power_divergence(f_obs=q, f_exp=p, lambda_="log-likelihood")
			# print('Power Divergence is: ', kl_div)
			# print('P value is: ', p_val)
		except FloatingPointError as e:
			print('Infinity')

		
		pow_div1 = round(pow_div1, 4)
		pow_div2 = round(pow_div2, 4)
		pow_div3 = round(pow_div3, 4)
		pow_div4 = round(pow_div4, 4)
		pow_div5 = round(pow_div5, 4)
		js_div = round(js_div, 4)
		p_div1.append(pow_div1)
		p_div2.append(pow_div2)
		p_div3.append(pow_div3)
		p_div4.append(pow_div4)
		p_div5.append(pow_div5)

		j_div.append(js_div)

		data_dict = {'Exponent':lambdas[ind],'JS Divergence':js_div, 'Lambda:2/3':pow_div1, 'Lambda:-2':pow_div2, \
		'Lambda:-1/2':pow_div3, 'Lambda:-1':pow_div4, 'Lambda:0':pow_div5}
		df = df.append(data_dict, ignore_index=True)

	return p_div1, p_div2, p_div3, p_div4, p_div5, j_div


def plot(dis):
	global df
	plt.style.use('seaborn-darkgrid')

	p_div1, p_div2, p_div3, p_div4, p_div5, j_div = calc_div(dis)
	
	plt.plot(lambdas, p_div1, label='Lambda (2/3)')
	plt.plot(lambdas, p_div5, label='Lambda (0)')
	plt.plot(lambdas, p_div3, label='Lambda (-1/2)')
	plt.plot(lambdas, p_div4, label='Lambda (-1)')
	# plt.plot(lambdas, p_div2, label='Lambda (-2)')
	plt.plot(lambdas, j_div, label='JS Divergence')

	y_ticks = np.arange(0, 2.4, 0.4)
	plt.yticks(y_ticks)
	plt.legend(fontsize=9)
	plt.title('Test-statistics: '+str(dis)+ ' diseases')
	plt.xlabel('Exponent (for exponential distribution)')
	plt.ylabel('Divergence')
	plt.show()

	print('DataFrame is:\n', df)

	df.to_csv('../output/expt1.1.1/divergence_d'+str(dis)+'.csv', index=False)

# globally accessible variables
diseases = 15
# lambdas = [1.25,0.63,0.42]
lambdas = [1.25, 0.83,0.63,0.5,0.42]
kls = {}
cols = ['Exponent', 'JS Divergence', 'Lambda:2/3', 'Lambda:0', 'Lambda:-1/2' , 'Lambda:-1', 'Lambda:-2']
df = pd.DataFrame(columns=cols)
plot(diseases)