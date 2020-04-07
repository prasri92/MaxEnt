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

np.seterr(all='raise')
# np.seterr(all='warn')

# Read the maxent prob. distribution for sum of diseases
def read_maxent_prob(filename):
	with open(filename, "rb") as outfile:
		data = pickle.load(outfile,encoding='latin1')
	return data[0], data[1], data[2]

# Read the true prob. distribution for sum of diseases
def read_true_prob(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[0], prob[1]

# Get divergence of probability distributions
def divergence(p, q):
	return (p*np.log(p/q)).sum()

def calc_div(k,f,ds_num):
	maxent_file = '../output_s'+str(ds_num)+'/d'+str(k)+'/syn_maxent_expt'+str(f)+'_pr2.pickle'
	maxent_dist, maxent_prob, emp_prob = read_maxent_prob(maxent_file)

	true_file = '../output_s'+str(ds_num)+'/d'+str(k)+'/truedist_expt'+str(f)+'.pickle'
	true_dist, true_prob = read_true_prob(true_file)

	p = np.array(true_dist)
	q = np.array(maxent_dist)

	
	try:
		kl_div = divergence(q, p)
	except:
		kl_div = 'NaN'

	js_div = distance.jensenshannon(p, q)
	try:
		pow_div, p_val = power_divergence(f_obs=q, f_exp=p, lambda_="cressie-read")
	except:
		pow_div = 'NaN'
	
	return kl_div, js_div, pow_div

if __name__ == '__main__':
	# # write to file
	div_file = 'outfiles/robust_divergence.csv'
	with open(div_file, "w",newline='') as csvFile: 
		first_row = ['index','#_diseases','dataset_num','file_num','kl_div','js_div','pow_div']
		csv.writer(csvFile).writerow(first_row)
		counter = 0
		for ds_num in range(30,36):
			for dis in [4,7,10,15]:
				for file_num in [1,2,3,4,5]:
					kl_div, js_div, pow_div = calc_div(f=file_num, k=dis, ds_num=ds_num)
					second_row = [counter, dis, ds_num, file_num,kl_div, js_div, pow_div]
					csv.writer(csvFile).writerow(second_row)
					counter += 1
	csvFile.close()