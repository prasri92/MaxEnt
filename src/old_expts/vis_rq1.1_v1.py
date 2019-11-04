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

np.seterr(all='raise')
lambdas = [0.42, 0.5, 0.63, 0.83, 1.25]
kls = []

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

def calc_kl(i,k,j):
	true_file = '../output/d'+str(k)+'/truedist_expt'+str(i)+'.pickle'
	true_dist, true_prob = read_true_prob(true_file)

	maxent_file = '../output/d'+str(k)+'_expt1.1/syn_maxent_expt'+str(i)+'_s'+str(j)+'.pickle'
	maxent_dist, maxent_prob, emp_prob, num_constraints, support = read_maxent_prob(maxent_file)

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
	return kl_div

def plot(file_num, dis_num, supports):
	print('Support, KL Divergence')
	for j,s in enumerate(supports):
		kls.append(calc_kl(file_num, dis_num, j=j))

	plt.style.use('seaborn-darkgrid')
	plt.plot(supports, kls, marker='o')
	plt.legend(fontsize=9)
	plt.title('Maximum Entropy vs. Support')
	plt.xlabel('Support')
	plt.ylabel('KL Divergence')
	plt.show()

def find_possible_files(file_num, dis_num):
	supports = []
	num_constraints = []
	for x in range(0, 20):
		maxent_file = '../output/d'+str(dis_num)+'_expt1.1/syn_maxent_expt'+str(file_num)+'_s'+str(x)+'.pickle'
		# if os.path.exists('../output/d'+str(dis_num)+'_expt1.1/syn_maxent_expt'+str(file_num)+'_s'+str(x)+'.pickle')==True:
		if os.path.exists(maxent_file)==True:
			maxent_dist, maxent_prob, emp_prob, num_constraint, support = read_maxent_prob(maxent_file)
			supports.append(support)
			num_constraints.append(num_constraint)

	return num_constraints, supports 

def write_csv(file_num, dis_num, supports, num_constraints):
	supports = np.around(supports, 3)
	# To better explain, write answers to CSV to show where the anomalies lie 
	df = pd.DataFrame({'Support':supports,
						'Constraints':num_constraints,
						'KL Divergence':kls})
	df.to_csv('../output/expt1.1/d'+str(dis_num)+'_l'+str(lambdas[int(file_num)])+'.csv', index=False)

dis_num = sys.argv[1]
file_num = sys.argv[2]	 

num_constraints, supports = find_possible_files(file_num, dis_num)
plot(file_num, dis_num, supports)
write_csv(file_num, dis_num, supports, num_constraints)