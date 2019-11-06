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
from sklearn.metrics import log_loss
import random

np.seterr(all='raise')

# Read the true prob. distribution for sum of diseases
def read_true_prob(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[0], prob[1]

# Get kl divergence of probability distributions
def kl_divergence(p, q):
	return (p*np.log(p/q)).sum()

def calc_kl(k, file_num):
	global df 
	kl = []
	for ind, l in enumerate(lambdas):
		true_file = '../output_s'+str(file_num)+'/d'+str(k)+'/truedist_expt'+str(ind+1)+'.pickle'
		true_dist, true_prob = read_true_prob(true_file)

		comp_file = '../output_s20/d'+str(k)+'/truedist_expt'+str(ind+1)+'.pickle'
		comp_dist, comp_prob = read_true_prob(comp_file)
		
		true_prob = np.around(true_prob, decimals=4)
		comp_prob = np.around(comp_prob, decimals=4)

		# if k==10 and ind==4:
			# print(random.sample(list(zip(true_dist, comp_dist)), 20))
			# print(true_prob)
			# print(comp_prob)

		p = np.array(true_dist)
		q = np.array(comp_dist)

		try:
			kl_div = kl_divergence(p, q)
		except FloatingPointError as e:
			print('Infinity')

		kl_div = round(kl_div, 4)
		kl.append(kl_div)

		data_dict = {'Lambda':lambdas[ind],'# Diseases':k, 'KL Divergence':kl_div}
		df = df.append(data_dict, ignore_index=True)

	return kl


def plot(file_num):
	global df
	kls = {}
	plt.style.use('seaborn-darkgrid')

	for ind, dis in enumerate(diseases):
		kls[dis] = calc_kl(dis, file_num)
		plt.plot(lambdas, kls[dis], label=str(dis)+' diseases')

	y_ticks = np.arange(-0.3, 1, 0.1)
	plt.yticks(y_ticks)
	plt.legend(fontsize=9)
	plt.title('Variation in zipfian (4.0 to 0.0) and binomial (0.75 to 0.5)')
	plt.xlabel('Lambda (Exponential Distribution)')
	plt.ylabel('KL Divergence')
	plt.show()

	# print(df)
	df.to_csv('../output_s10/expt_3.1.csv', index=False)

# globally accessible variables
diseases = [4,7,10,15]
lambdas = [0.42, 0.5, 0.63, 0.83, 1.25]
kls = {}
cols = ['Lambda', '# Diseases', 'KL Divergence']
for f in range(1,20):
	df = pd.DataFrame(columns=cols)
	print('File_number: ', f)
	plot(f)
	print()

