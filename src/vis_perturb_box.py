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
# Read the true prob. distribution for sum of diseases
def read_true_prob(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[1]

# Read the perturbed prob. and regularization prob. distribution 
def read_perturb_prob(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[0], prob[1], prob[2], prob[3]

# Get kl divergence of probability distributions
def kl_divergence(p, q):
	return (p*np.log(p/q)).sum()

def compute_kl(i, size, num_feats, perturb_prob):
	'''
	Compute the kl divergence among the exact maxent and approximate methods 
	'''
	true_file = '../output/d'+str(size)+'_'+str(num_feats)+'/truedist_expt'+str(i)+'.pickle'
	true_prob = read_true_prob(true_file)

	perturb_file = '../output/d'+str(size)+'_'+str(num_feats)+'_perturb_box/syn_maxent_expt'+str(i)+'_'+str(perturb_prob)+'.pickle'
	m_dist_p, m_prob_p, m_dist_r, m_prob_r = read_perturb_prob(perturb_file)

	kl_div = []
	for i in range(len(m_dist_r)):
		p = (np.array(m_dist_p))
		q = (np.array(m_dist_r[i]))
		kl_div.append(kl_divergence(q,p))

	return true_prob, m_prob_p, m_prob_r, kl_div

def plot(file_num, num_feats, size):
	true = []
	perturb = []
	reg = []
	kl = []

	p_prob = [0.01, 0.04, 0.1, 0.2]

	for p in p_prob:
		t, p, r, k = compute_kl(file_num, size, num_feats, p)
		true.append(t)
		perturb.append(p)
		reg.append(r)
		kl.append(k)

		# print("The KL divergences are: ", k)

	num_feats = num_feats
	xvec = [i for i in range(num_feats+1)]
	x_ticks = np.arange(0, num_feats+1, 1.0)
	plot_lims = [-0.3,  num_feats+0.3, -0.1, 1.0]

	plt.style.use('seaborn-darkgrid')

	fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(30,10))
	lst = [ax0, ax1, ax2, ax3]

	# width = [0.01, 0.05, 0.1, 0.3]
	width = [0.05, 0.1, 0.5, 0.8]

	for num,i in enumerate(lst):
		perturb[num] = np.around(perturb[num], decimals=4)
		true[num] = np.around(true[num], decimals=4)

		i.plot(xvec, true[num], color='k',  label='True Distribution ',)
		i.plot(xvec, perturb[num][0], linestyle='solid', label='Maxent Pert. Unreg.')
		
		best_w = kl[num].index(min(kl[num]))
		for w in range(len(width)):
			if w == best_w:
				i.plot(xvec, reg[num][w], label=str(width[w])+' , KL Div: '+str(np.around(kl[num][w], decimals=4)))
			else:
				i.plot(xvec, reg[num][w], linestyle='dotted', label=str(width[w])+' , KL Div: '+str(np.around(kl[num][w], decimals=4)))
			
		i.axis(plot_lims)
		i.set_xticks(x_ticks)
		# i.xlabel("Number of diseases per patient")
		# i.ylabel("Prob.")
		# for xy in zip(xvec, maxent_prob[num]):
		# 	i.annotate(('%s, %s') %xy, xy=xy, textcoords='offset points')
		if num==0:
			i.set_title('Perturbed Prob = ' + str(p_prob[num]), fontsize=10)
		elif num==1:
			i.set_title('Perturbed Prob = ' + str(p_prob[num]), fontsize=10)
		elif num==2:
			i.set_title('Perturbed Prob = ' + str(p_prob[num]), fontsize=10)
		elif num==3:
			i.set_title('Perturbed Prob = ' + str(p_prob[num]), fontsize=10)

		i.legend(fontsize=9)
		row = []

	if file_num == 3:
		fig.suptitle('Diseases = '+str(num_feats)+', Dataset Size = '+str(size)+'\nLambda = 1.25, Skew = 2', y=0.99, fontsize=10)
	elif file_num == 13:
		fig.suptitle('Diseases = '+str(num_feats)+', Dataset Size = '+str(size)+'\nLambda = 0.416, Skew = 2', y=0.99, fontsize=10)
	elif file_num == 23:
		fig.suptitle('Diseases = '+str(num_feats)+', Dataset Size = '+str(size)+'\nLambda = 0.25, Skew = 2', y=0.99, fontsize=10)
	# plt.subplots_adjust(hspace = 0.6, top=0.85)
	plt.subplots_adjust(top=0.85)
	plt.show()

if __name__ == '__main__':
	i = sys.argv[1]
	num_feats = sys.argv[2]
	size = sys.argv[3]
	plot(int(i), int(num_feats), int(size))