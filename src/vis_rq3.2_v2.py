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
from scipy.spatial import distance
from scipy.stats import power_divergence


np.seterr(all='raise')

# Read the true prob. distribution for sum of diseases
def read_true_prob(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[0], prob[1]

def read_max_prob(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[0], prob[1], prob[2], prob[3]

# Get kl divergence of probability distributions
# def kl_divergence(p, q):
# 	return (p*np.log(p/q)).sum()

def calc_kl(k, ds_num=None):
	kl_emp_p = []
	kl_max_p = []

	for i in range(1,6):
		if ds_num == None:
			true_d, true_p = read_true_prob('../output/d'+str(k)+'/truedist_expt'+str(i)+'.pickle')
			max_d, max_p, emp_d, emp_p = read_max_prob('../output/d'+str(k)+'_expt3.2_rob_ls/syn_emp_expt'+str(i)+'.pickle')
		else:
			true_d, true_p = read_true_prob('../output_s'+str(ds_num)+'/d'+str(k)+'/truedist_expt'+str(i)+'.pickle')
			max_d, max_p, emp_d, emp_p = read_max_prob('../output_s'+str(ds_num)+'/d'+str(k)+'_expt3.2_rob_ls/syn_emp_expt'+str(i)+'.pickle')

		p = np.array(true_d)
		q = np.array(max_d)
		r = np.array(emp_d)

		# print('True Distribution', p)
		# print('MaxEnt Distribution', q)
		# print('Empirical Distribution', r)
		
		try:
			kl_1, p_val_1 = power_divergence(f_obs=r, f_exp=p, lambda_="cressie-read")
			kl_2, p_val_2 = power_divergence(f_obs=q, f_exp=p, lambda_="cressie-read")
			# kl_1 = distance.jensenshannon(p, q)
			# kl_2 = distance.jensenshannon(p, r)
		except:
			p[p == 0] = 1e-300
			kl_1, p_val_1 = power_divergence(f_obs=r, f_exp=p, lambda_="cressie-read")
			kl_2, p_val_2 = power_divergence(f_obs=q, f_exp=r, lambda_="cressie-read")

		kl_emp_p.append(kl_1)
		kl_max_p.append(kl_2)

		
	return kl_emp_p, kl_max_p 

# def plot(k, ds_num):
# 	kl1, kl2 = calc_kl(k, ds_num)
# 	plt.style.use('seaborn-darkgrid')
# 	lambdas = [1.25, 0.83, 0.63, 0.50, 0.42]

# 	# print('Empirical: ', kl1)
# 	# print('Maxent: ',kl2)

# 	# y_ticks = np.arange(0, 3, 0.5)
# 	plt.plot(lambdas, kl1, marker='o', label='Empirical')
# 	plt.plot(lambdas, kl2, marker='d', label='Maxent')

# 	plt.legend(fontsize=9)
# 	# plt.yticks(y_ticks)
# 	plt.title('Empirical vs. MaxEnt: '+str(k)+' diseases \n Robust optimizer with Learned Support')
# 	plt.xlabel('Exponent')
# 	plt.ylabel(r'Power Divergence ($\lambda$ = 2/3)')
# 	plt.savefig('../figures/Experiments/emp_d'+str(k)+'_rob_ls.png', format='png')
# 	plt.show()

# def plot_main(k):
# 	kl1, kl2 = calc_kl(k)
# 	plt.style.use('seaborn-darkgrid')
# 	lambdas = [1.25, 0.83, 0.63, 0.50, 0.42]

# 	# print('Empirical: ', kl1)
# 	# print('Maxent: ',kl2)

# 	# y_ticks = np.arange(0, 3, 0.5)
# 	plt.plot(lambdas, kl1, marker='o', label='Empirical')
# 	plt.plot(lambdas, kl2, marker='d', label='Maxent')

# 	plt.legend(fontsize=9)
# 	# plt.yticks(y_ticks)
# 	plt.title('Empirical vs. MaxEnt: '+str(k)+' diseases\n Robust optimizer with Learned Support')
# 	plt.xlabel('Exponent')
# 	plt.ylabel(r'Power Divergence ($\lambda$ = 2/3)')
# 	plt.savefig('../figures/Experiments/emp_d'+str(k)+'_rob_ls.png', format='png')
# 	plt.show()

# num_dis = sys.argv[1]
# if len(sys.argv) > 2:
# 	ds_num = sys.argv[2]
# 	plot(int(num_dis), int(ds_num))
# else:
# 	plot_main(int(num_dis))

def plot():
	plt.style.use('seaborn-darkgrid')
	plt.rcParams['font.size'] = 10
	plt.rcParams['axes.labelsize'] = 10
	plt.rcParams['axes.labelweight'] = 'bold'
	plt.rcParams['axes.titlesize'] = 10
	plt.rcParams['xtick.labelsize'] = 8
	plt.rcParams['ytick.labelsize'] = 8
	plt.rcParams['legend.fontsize'] = 10
	plt.rcParams['figure.titlesize'] = 12

	lambdas = [1.25, 0.83, 0.63, 0.50, 0.42]
	fig, axes = plt.subplots(2,2)
	
	y_ticks = np.arange(0, 35, 10)
	idx_mapping = {0:[0,0], 1:[0,1], 2:[1,0], 3:[1,1]}
	plt.setp(axes, yticks=y_ticks)

	for idx, k in enumerate([4,7,10,15]):
		kl1, kl2 = calc_kl(k)
		i,j = idx_mapping[idx]
		axes[i,j].plot(lambdas, kl1, marker='o', label='Empirical')
		axes[i,j].plot(lambdas, kl2, marker='d', label='MaxEnt_MCC')
		axes[i,j].set_title(str(k) + ' diseases')
		axes[i,j].set_ylim([-5, 35])
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

	fig.text(0.5, 0.07, r'Exponent($\alpha$)', ha='center')
	fig.text(0.07, 0.5, r'Power Divergence ($\lambda$ = 2/3)', va='center', rotation='vertical')
	fig.suptitle('Empirical distribution vs. MaxEnt_MCC distribution')
	plt.show()

plot()