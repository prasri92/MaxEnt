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

# np.seterr(divide='raise', over='raise', under='raise')
np.seterr(all='raise')

# Read the maxent prob. distribution for sum of diseases
def read_maxent_prob(filename):
	try:
		with open(filename, "rb") as outfile:
			data = pickle.load(outfile,encoding='latin1')
		return (data[0], data[1], data[2], data[3], data[4])
	except IOError as e:
		return 'NaN'
	except EOFError as e:
		return 'NaN'

# Read the true prob. distribution for sum of diseases
def read_true_prob(filename):
	try:
		with open(filename, "rb") as outfile:
			prob = pickle.load(outfile,encoding='latin1')
		return (prob[0], prob[1])
	except IOError as e:
		return 'NaN'
	except EOFError as e:
		return 'NaN'

# Get divergence of probability distributions
def divergence(p, q):
	return (p*np.log(p/q)).sum()

def calc_div(k,f,ds_num,sup_num):
	"""
	Calculate all 3 types of divergence between the corresponding maxent distribution
	and the true distribution. 
	Write into another file that outlines each dataset and each disease
	"""
	maxent_file = '../output/output_s'+str(ds_num)+'/d'+str(k)+'_expt1.2/syn_maxent_expt'+str(f)+'_s'+str(sup_num)+'.pickle'
	maxent_output = read_maxent_prob(maxent_file)
	# print(maxent_output)
	if maxent_output == 'NaN':
		return "NaN"
	else:
		maxent_dist, maxent_prob, emp_prob, cons, support = maxent_output

	true_file = '../output/output_s'+str(ds_num)+'/d'+str(k)+'/truedist_expt'+str(f)+'.pickle'
	true_dist, true_prob = read_true_prob(true_file)
	
	p = np.array(true_dist)
	q = np.array(maxent_dist)

	try:
		kl_div = divergence(q, p)
	except ValueError as e: 
		kl_div = 'NaN'
	except:
		kl_div = 'NaN'

	# js_div = distance.jensenshannon(p, q)
	try:
		js_div = distance.jensenshannon(p, q)
	except FloatingPointError as e:
		# print('K is: ', str(k), ' File: ', str(f), ' DS Num: ', str(ds_num))
		q[q < 1e-300] = 0.0
		# print('Shape of p:', p.shape)
		# print('Shape pf q:', q.shape)
		# print("Sum(q)", np.sum(q,axis=0))
		js_div = distance.jensenshannon(p,q)
	except ValueError as e:
		# print('Shape of p:', p.shape)
		# print('Shape pf q:', q.shape)
		js_div = 'NaN'

	try:
		pow_div, p_val = power_divergence(f_obs=q, f_exp=p, lambda_="cressie-read")
	except ValueError as e: 
		pow_div = 'NaN'
	except: 
		pow_div = 'NaN'

	return (kl_div, js_div, pow_div, cons+k+1, support) 

if __name__ == '__main__':
	# # write to file
	sup_file = '../support_analysis/support.csv'
	# with open(sup_file, "w",newline='') as csvFile: 

	# append to the same file 
	with open(sup_file, "a") as csvFile:
		counter = 86400
		for ds_num in range(81,101):
			for dis in range(4,21,2):
				for file_num in range(1,11):
					for sup_num in range(1,13):
						out = calc_div(f=file_num, k=dis, ds_num=ds_num, sup_num=sup_num)
						if out != 'NaN':
							kl_div, js_div, pow_div, constraints, support = out 
							second_row = [counter, dis, ds_num, file_num, sup_num, constraints, support, kl_div, js_div, pow_div]
							csv.writer(csvFile).writerow(second_row)
							counter += 1
						else:
							second_row = [counter, dis, ds_num, file_num, sup_num, '-', '-', '-', '-', '-']
							csv.writer(csvFile).writerow(second_row)
							counter += 1 
	csvFile.close()
	'''
	with open(sup_file, "w",) as csvFile: 
		first_row = ['index','#_diseases','dataset_num','file_num','sup_num','constraints','support','kl_div','js_div','pow_div']
		csv.writer(csvFile).writerow(first_row)
		counter = 0
		for ds_num in range(1,101):
			for dis in range(4,21,2):
				for file_num in range(1,11):
					for sup_num in range(1,13):
						out = calc_div(f=file_num, k=dis, ds_num=ds_num, sup_num=sup_num)
						if out != 'NaN':
							kl_div, js_div, pow_div, constraints, support = out 
							second_row = [counter, dis, ds_num, file_num, sup_num, constraints, support, kl_div, js_div, pow_div]
							csv.writer(csvFile).writerow(second_row)
							counter += 1
						else:
							second_row = [counter, dis, ds_num, file_num, sup_num, '-', '-', '-', '-', '-']
							csv.writer(csvFile).writerow(second_row)
							counter += 1
	csvFile.close()
	'''