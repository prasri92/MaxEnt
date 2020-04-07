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

	maxent_ur_up, maxent_r_up = read_up_prob('../output/d'+str(k)+'_expt2.3/syn_maxent_up'+str(i)+'.pickle')

	for p in perturb_prob:
		maxent_file = '../output/d'+str(k)+'_expt2.3/syn_maxent_p'+str(i)+'_'+str(p)+'.pickle'
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

def plot(k):
	# x_ticks = np.arange(0, 0.23, 0.02)
	# y_ticks = np.arange(0, 3, 0.5)
	plt.style.use('seaborn-darkgrid')

	kl_ur_p, kl_r_p = calc_kl(k=k)

	plt.plot(perturb_prob, kl_ur_p, marker='o', label='Unregularized')
	plt.plot(perturb_prob, kl_r_p, marker='d', label='Regularized')

	# plt.xticks(x_ticks)
	# plt.yticks(y_ticks)
	plt.legend(fontsize=9)
	plt.title('Regularization for different Perturbations: '+str(k)+' diseases (Chosen Support)\n'+'95% CI Width')
	plt.xlabel('Perturbed Probability')
	plt.ylabel(r'Power Divergence ($\lambda$ = 2/3)')
	# plt.ylabel('Jensen-Shannon Divergence')
	plt.savefig('../figures/expt2.3/dis_'+str(k)+'_chosensup.png')
	plt.show()

num_dis = sys.argv[1]
plot(int(num_dis))
