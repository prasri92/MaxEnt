'''
Python = 3.7 
matplotlib to plot figures for all diseases together
'''
#PYTHON3 
import numpy as np
import pickle 
import matplotlib.pyplot as plt
import csv
import pandas as pd
import sys
from scipy.stats import power_divergence
from scipy.spatial import distance


np.seterr(all='raise')
perturb_prob = [0.01, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21]
# perturb_prob = [0.01, 0.03, 0.06, 0.09, 0.12, 0.15]

# Read the true prob. distribution for sum of diseases
def read_up_prob(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[0], prob[1]

def read_p_prob(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[0], prob[1]

# Get kl divergence of probability distributions
def kl_divergence(p, q):
	return (p*np.log(p/q)).sum()

def calc_kl(k, i=3):
	kl_ur_p = []
	kl_r_p = []

	maxent_ur_up, maxent_r_up = read_up_prob('../output/d'+str(k)+'_expt2.1_w1_ls/syn_maxent_up'+str(i)+'.pickle')

	for p in perturb_prob:
		maxent_file = '../output/d'+str(k)+'_expt2.1_w1_ls/syn_maxent_p'+str(i)+'_'+str(p)+'.pickle'
		maxent_ur_p, maxent_r_p = read_p_prob(maxent_file)

		p = np.array(maxent_ur_up)
		q = np.array(maxent_ur_p)
		r = np.array(maxent_r_up)
		s = np.array(maxent_r_p)
		try:
			kl_1, p_val_1 = power_divergence(f_obs=q, f_exp=p, lambda_="cressie-read")
			kl_2, p_val_2 = power_divergence(f_obs=s, f_exp=r, lambda_="cressie-read")
			# kl_1 = distance.jensenshannon(p, q)
			# kl_2 = distance.jensenshannon(r, s)
			# kl_1 = kl_divergence(p, q)
			# kl_2 = kl_divergence(r, s)
		except:
			p[p == 0] = 1e-300
			r[r == 0] = 1e-300
			kl_1, p_val_1 = power_divergence(f_obs=q, f_exp=p, lambda_="cressie-read")
			kl_2, p_val_2 = power_divergence(f_obs=s, f_exp=r, lambda_="cressie-read")
			# print('P:', p)
			# print('Q:', q)
			# print('R:', r)
			# print('S:', s)
			

		kl_ur_p.append(kl_1)
		kl_r_p.append(kl_2)

	return kl_ur_p, kl_r_p

def plot():
	plt.style.use('seaborn-darkgrid')
	# plt.rcParams['font.family'] = 'serif'
	# plt.rcParams['font.serif'] = 'Ubuntu'
	# plt.rcParams['font.monospace'] = 'Ubuntu Mono'
	plt.rcParams['font.size'] = 10
	plt.rcParams['axes.labelsize'] = 10
	plt.rcParams['axes.labelweight'] = 'bold'
	plt.rcParams['axes.titlesize'] = 10
	plt.rcParams['xtick.labelsize'] = 8
	plt.rcParams['ytick.labelsize'] = 8
	plt.rcParams['legend.fontsize'] = 10
	plt.rcParams['figure.titlesize'] = 12
	fig, axes = plt.subplots(2,2)
	
	y_ticks = np.arange(0, 100, 10)
	idx_mapping = {0:[0,0], 1:[0,1], 2:[1,0], 3:[1,1]}
	plt.setp(axes, yticks=y_ticks)

	for idx, k in enumerate([4,7,10,15]):
		kl_ur_p, kl_r_p = calc_kl(k=k)
		i,j = idx_mapping[idx]
		axes[i,j].plot(perturb_prob, kl_ur_p, marker='o', label='Unregularized')
		axes[i,j].plot(perturb_prob, kl_r_p, marker='d', label='Regularized')
		axes[i,j].set_title(str(k) + ' diseases')
		axes[i,j].set_ylim([-5, 100])
		# axes[i,j].set_xlabel('Perturbed Probability')
		# axes[i,j].set_ylabel(r'Power Divergence ($\lambda$ = 2/3)')
		axes[i,j].set_yticks(y_ticks)
		axes[i,j].legend(loc='best', numpoints=1, fancybox=True, fontsize=9)

	# plt.xticks(x_ticks)
	# plt.yticks(y_ticks)
	# plt.legend(fontsize=9)
	# plt.title('Regularization for different Perturbations: (Learned Support)\n'+'Single Width W = 1')
	# plt.xlabel('Perturbed Probability')
	# plt.ylabel(r'Power Divergence ($\lambda$ = 2/3)')
	# plt.ylabel('Jensen-Shannon Divergence')
	# plt.savefig('../figures/Experiments/pert_d'+str(k)+'_robust_ls.png')

	fig.text(0.5, 0.07, 'Perturbed Probability', ha='center')
	fig.text(0.07, 0.5, r'Power Divergence ($\lambda$ = 2/3)', va='center', rotation='vertical')
	fig.suptitle('Regularization for different Perturbations: (Learned Support)\n'+'Single Width W = 1')
	plt.show()

plot()