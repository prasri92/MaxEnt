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
from scipy.stats import power_divergence
from scipy.spatial import distance

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

# Get power divergence of probability distributions
def divergence(p, q):
	# div = distance.jensenshannon(p,q)
	div, _ = power_divergence(f_obs=q, f_exp=p, lambda_="freeman-tukey")
	return div
	# return (p*np.log(p/q)).sum()

def calc_div(i,k,dataset_num=None):
	constraints = []
	sup_metric = []
	divs = []
	global df 
	for j in range(12):
		if dataset_num == None:
			true_file = '../output/d'+str(k)+'/truedist_expt'+str(i)+'.pickle'
			maxent_file = '../output/d'+str(k)+'_expt1.2/syn_maxent_expt'+str(i)+'_s'+str(j)+'.pickle'
		else:
			true_file = '../output_s'+dataset_num+'/d'+str(k)+'/truedist_expt'+str(i)+'.pickle'
			maxent_file = '../output_s'+dataset_num+'/d'+str(k)+'_expt1.2/syn_maxent_expt'+str(i)+'_s'+str(j)+'.pickle'
		
		maxent_dist, maxent_prob, emp_prob, num_constraints, support = read_maxent_prob(maxent_file)
		true_dist, true_prob = read_true_prob(true_file)

		emp_prob = np.around(emp_prob, decimals=4)
		maxent_prob = np.around(maxent_prob, decimals=4)
		true_prob = np.around(true_prob, decimals=4)
		support = np.around(support, decimals=3)

		# While using the Jensen Shannon divergence allows us to ignore the divergence 
		# where the probabilities are zero. 
		# np.seterr(all='ignore')
		p = np.array(true_dist)
		q = np.array(maxent_dist)
		try:
			div = divergence(p,q)
			# kl_div = kl_divergence(p, q)
		except FloatingPointError as e:
			# CHECK IF FIX BY ZERO ATOM DETECTION 
			print(q)
			print(np.isfinite(q).all())
			# print(p)
			print(e)
			# print(q)
			# div = 0.0
			print('Divide by zero encountered')

		div = round(div, 4)
		constraints.append(num_constraints)
		sup_metric.append(support)
		divs.append(div)

		data_dict = {'Lambda':lambdas[i-1], 'Support':support, 'Constraints':num_constraints, \
			 'Power Divergence':div}
		df = df.append(data_dict, ignore_index=True)

	return constraints, sup_metric, divs


def plot(dis_num, dataset_num=None):
	global df
	plt.style.use('seaborn-darkgrid')

	for ind, l in enumerate(lambdas):
		cons[l], sups[l], div_vals[l] = calc_div(ind+1, dis_num, dataset_num)
		plt.plot(sups[l], div_vals[l], label='Lambda :'+str(l))

	plt.legend(fontsize=9)
	plt.title('Maxent vs. Support: '+str(dis_num)+' diseases')
	plt.xlabel('Support')
	plt.ylabel('Power Divergence')
	y_ticks = np.arange(0,0.5,0.1)
	plt.yticks(y_ticks)
	if dataset_num == None:
		plt.savefig('../figures/expt1.2/main_dis'+str(dis_num)+'_powdiv.png', format='png')
		df.to_csv('../output/expt1.2/d'+str(dis_num)+'_powdiv.csv', index=False)
	else:
		plt.savefig('../figures/expt1.2/dataset'+dataset_num+'_dis'+str(dis_num)+'.png', format='png')
		df.to_csv('../output_s'+dataset_num+'/expt1.2/d'+str(dis_num)+'.csv', index=False)
		print("Dataset", dataset_num, ' Diseases ', dis_num, 'Yes')

# globally accessible variables
lambdas = [1.25,0.83,0.63,0.5,0.42]
cons = {}
sups = {}
div_vals = {}
cols = ['Lambda', 'Support', 'Constraints', 'Power Divergence']
df = pd.DataFrame(columns=cols)

dis_num = sys.argv[1]
if len(sys.argv) > 2:
	dataset_num = sys.argv[2]
	plot(dis_num, dataset_num)
else:
	plot(dis_num)
