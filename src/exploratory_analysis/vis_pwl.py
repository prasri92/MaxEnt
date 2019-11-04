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

'''
Functions to read the probability distributions from stored files
'''
def read_prob_dist(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[0]

def read_exact_maxent(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[1], prob[2]

def read_true_prob(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[1]

def read_forcedp_prob(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[1]

def read_pwl_prob(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[1]

def kl_divergence(p, q):
	return (p*np.log(p/q)).sum()

def compute_kl(i, size, num_feats):
	'''
	Compute the kl divergence among the exact maxent and approximate methods 
	'''
	true_file = '../output/d'+str(size)+'_'+str(num_feats)+'/truedist_expt'+str(i)+'.pickle'
	exact_file = '../output/d'+str(size)+'_'+str(num_feats)+'_addzeros/syn_maxent_expt'+str(i)+'.pickle'
	emp_file = '../output/d'+str(size)+'_'+str(num_feats)+'_addzeros/syn_maxent_expt'+str(i)+'.pickle'
	forcedp_file = '../output/d'+str(size)+'_'+str(num_feats)+'_mu15/syn_maxent_expt'+str(i)+'.pickle'
	pwl_file = '../output/d'+str(size)+'_'+str(num_feats)+'_pwl/syn_maxent_expt'+str(i)+'.pickle'

	exact_prob, emp_prob = read_exact_maxent(exact_file)
	true_prob = read_true_prob(true_file)
	forcedp_prob = read_forcedp_prob(forcedp_file)
	pwl_prob = read_pwl_prob(pwl_file)

	p = (np.array(exact_prob))
	q = (np.array(pwl_prob))
	kl_div = kl_divergence(q,p)

	return true_prob, exact_prob, emp_prob, forcedp_prob, pwl_prob, kl_div

def plot(num_feats, size):
	exact = []
	emp = []
	kl = []
	true = []
	forcedp = []
	pwl = []
	kl = []

	for i in range(3, 24, 10):
		t,ex,em,f,pw,k = compute_kl(i, size, num_feats)
		true.append(t)
		exact.append(ex)
		emp.append(em)
		forcedp.append(f)
		pwl.append(pw)
		kl.append(k)

	num_feats = num_feats
	xvec = [i for i in range(num_feats+1)]
	x_ticks = np.arange(0, num_feats+1, 1.0)
	plot_lims = [-1,  num_feats+1, -0.1, 1.0]

	fig, (ax0, ax1, ax2) = plt.subplots(1,3, figsize=(15,4))
	lst = [ax0, ax1, ax2]

	start_file_num = 3
	for num,i in enumerate(lst):
		emp[num] = np.around(emp[num], decimals=4)
		exact[num] = np.around(exact[num], decimals=4)
		true[num] = np.around(true[num], decimals=4)
		forcedp[num] = np.around(forcedp[num], decimals=4)
		pwl[num] = np.around(pwl[num], decimals=4)

		# Write answers to CSV
		data = {'Empirical':emp[num], 'Maximum Entropy (Exact) Prob.':exact[num], 'True Prob.':true[num], \
		'Forced Partitioning': forcedp[num], 'Piecewise Likelihood':pwl[num]}
		df = pd.DataFrame(data=data)
		df.to_csv('../output/output_dist/all_comp_'+str(num_feats)+'dis_'+str(start_file_num)+'_file.csv')
		
		i.plot(xvec, exact[num], 'r', label='Maxent Exact')
		i.plot(xvec, emp[num], 'k', label='Empirical')
		# i.plot(xvec, true[num], 'm', label='True')
		i.plot(xvec, forcedp[num], 'g', label='Forced Part.')
		i.plot(xvec, pwl[num], 'b', label='Piecewise')

		i.axis(plot_lims)
		i.set_xticks(x_ticks)
		# i.xlabel("Number of diseases per patient")
		# i.ylabel("Prob.")
		# for xy in zip(xvec, maxent_prob[num]):
		# 	i.annotate(('%s, %s') %xy, xy=xy, textcoords='offset points')
		if num==0:
			i.set_title('KL div = ' + str(kl[num]) + '\nLambda = 1.25, Skew = 2', fontsize=7)
		elif num==1:
			i.set_title('KL div = ' + str(kl[num]) + '\nLambda = 0.416, Skew = 2', fontsize=7)
		elif num==2:
			i.set_title('KL div = ' + str(kl[num]) + '\nLambda = 0.25, Skew = 2', fontsize=7)

		i.legend(fontsize=6)
		row = []
		start_file_num += 10

	fig.suptitle('Diseases = '+str(num_feats)+', Dataset Size = '+str(size)+'\n', y=0.99, fontsize=10)
	# plt.subplots_adjust(hspace = 0.6, top=0.85)
	plt.subplots_adjust(top=0.85)
	plt.show()

if __name__ == '__main__':
	num_feats = sys.argv[1]
	size = sys.argv[2]
	plot(int(num_feats), int(size))