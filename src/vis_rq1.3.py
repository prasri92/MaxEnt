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
	maxent_file_zeros = '../output_s'+str(ds_num)+'/d'+str(k)+'/syn_maxent_z_expt'+str(f)+'.pickle'
	maxent_dist_z, maxent_prob_z, emp_prob_z = read_maxent_prob(maxent_file_zeros)

	maxent_file = '../output_s'+str(ds_num)+'/d'+str(k)+'/syn_maxent_expt'+str(f)+'.pickle'
	maxent_dist, maxent_prob, emp_prob = read_maxent_prob(maxent_file)

	true_file = '../output_s'+str(ds_num)+'/d'+str(k)+'/truedist_expt'+str(f)+'.pickle'
	true_dist, true_prob = read_true_prob(true_file)

	p = np.array(true_dist)
	q1 = np.array(maxent_dist_z)
	q2 = np.array(maxent_dist)

	# print(p)
	# print(q1)
	# print(q2)
	try:
		kl_div1 = divergence(q1, p)
	except:
		kl_div1 = 'NaN'
	try:
		kl_div2 = divergence(q2, p)
	except:
		kl_div2 = 'NaN'
	# print(kl_div1)
	# print(kl_div2)

	js_div1 = distance.jensenshannon(p, q1)
	js_div2 = distance.jensenshannon(p, q2)
	# print(js_div1)
	# print(js_div2)
	try:
		pow_div1, p_val = power_divergence(f_obs=q1, f_exp=p, lambda_="cressie-read")
	except:
		pow_div1 = 'NaN'
	try:
		pow_div2, p_val = power_divergence(f_obs=q2, f_exp=p, lambda_="cressie-read")
	except:
		pow_div2 = 'NaN'
	# print(pow_div1)
	# print(pow_div2)
	return kl_div1, kl_div2, js_div1, js_div2, pow_div1, pow_div2

if __name__ == '__main__':
	# # write to file
	div_file = '../zero_compare/divergence.csv'
	with open(div_file, "w",newline='') as csvFile: 
		first_row = ['index','#_diseases','dataset_num','file_num','kl_div_z','kl_div','js_div_z','js_div','pow_div_z','pow_div']
		csv.writer(csvFile).writerow(first_row)
		counter = 0
		for ds_num in [10, 20, 30, 40, 50, 51,52,53,54,55,56,57,58,59, 60]:
			for dis in [15]:
				for file_num in [1,2,3,4,5]:
					kl_div_z, kl_div, js_div_z, js_div, pow_div_z, pow_div = calc_div(f=file_num, k=dis, ds_num=ds_num)
					second_row = [counter, dis, ds_num, file_num, kl_div_z, kl_div, js_div_z, js_div, pow_div_z, pow_div]
					csv.writer(csvFile).writerow(second_row)
					counter += 1
	csvFile.close()