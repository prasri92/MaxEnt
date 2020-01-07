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
import random
from scipy.stats import power_divergence

np.seterr(all='ignore')

# Read the true prob. distribution for sum of diseases
def read_true_prob(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[0], prob[1]

# Get power divergence of probability distributions
def get_div(p, q):
	try:
		pow_div, p_val = power_divergence(f_obs=q, f_exp=p, lambda_="freeman-tukey")
	except FloatingPointError as e:
		print('If error, caused by - ', e)
		pow_div = np.nan
	return pow_div 

def calc_div(k, file_num):
	global df 
	divs = []
	for ind, l in enumerate(lambdas):
		true_file = '../output_s1/d'+str(k)+'/truedist_expt'+str(ind+1)+'.pickle'
		true_dist, true_prob = read_true_prob(true_file)

		comp_file = '../output_s'+str(file_num)+'/d'+str(k)+'/truedist_expt'+str(ind+1)+'.pickle'
		comp_dist, comp_prob = read_true_prob(comp_file)
		
		true_prob = np.around(true_prob, decimals=4)
		comp_prob = np.around(comp_prob, decimals=4)

		p = np.array(true_dist)
		q = np.array(comp_dist)
		#call divergence method 
		div = get_div(p,q)
		div = round(div, 4)
		divs.append(div)

		data_dict = {'Exponent':lambdas[ind],'# Diseases':k, 'Power Divergence':div}
		df = df.append(data_dict, ignore_index=True)

	return divs

def plot(file_num, title_txt):
	global df
	divergences = {}
	plt.style.use('seaborn-darkgrid')

	for ind, dis in enumerate(diseases):
		divergences[dis] = calc_div(dis, file_num)
		plt.plot(lambdas, divergences[dis], label=str(dis)+' diseases')

	y_ticks = np.arange(0.0, 2., 0.4)
	plt.yticks(y_ticks)
	plt.legend(fontsize=9)
	plt.title(title_txt)
	plt.xlabel('Exponent (Exponential Distribution)')
	plt.ylabel('Power Divergence')
	plt.savefig('../figures/expt3.1/dataset'+str(file_num)+'.png', format='png')

	print(df)
	df.to_csv('../output_expt3.1/f_'+str(file_num)+'.csv', index=False)

# globally accessible variables
diseases = [4,7,10,15]
lambdas = [1.25, 0.83, 0.63, 0.50, 0.42]
divergences = {}
cols = ['Exponent', '# Diseases', 'Power Divergence']
file_txt = {26:'Variation in q1 (0.75 to 0.25) and cluster configuration', \
			28:'Variation in q1 (0.75 to 0.50) and cluster configuration', \
			30:'Variation in cluster configuration', \
			32:'Variation in q1 (0.75 to 0.25), z (0.0 to 2.0) \n and cluster configuration', \
			34:'Variation in q1 (0.75 to 0.50), z (0.0 to 2.0) \n and cluster configuration', \
			36:'Variation in z (0.0 to 2.0) and cluster configuration', \
			38:'Variation in q1 (0.75 to 0.25), z (0.0 to 4.0) \n and cluster configuration', \
			40:'Variation in q1 (0.75 to 0.25), z (0.0 to 4.0) \n and cluster configuration', \
			42:'Variation in z (0.0 to 4.0) and cluster configuration', \
			44:'Variation in q1 (0.75 to 0.25)', \
			46:'Variation in q1 (0.75 to 0.50)', \
			48:'No variation in parameters', \
			50:'Variation in q1 (0.75 to 0.25), z (0.0 to 2.0)', \
			52:'Variation in q1 (0.75 to 0.50), z (0.0 to 2.0)', \
			54:'Variation in z (0.0 to 2.0)', \
			56:'Variation in q1 (0.75 to 0.25), z (0.0 to 4.0)', \
			58:'Variation in q1 (0.75 to 0.25), z (0.0 to 4.0)', \
			60:'Variation in z (0.0 to 4.0)'
			}
for f in range(26, 62, 2):
	df = pd.DataFrame(columns=cols)
	plt.figure()
	plot(f, file_txt[f])
	print()
