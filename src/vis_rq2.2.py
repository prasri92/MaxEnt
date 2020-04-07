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

np.seterr(all='raise')
perturb_prob = [0.01, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21]
# width = [0.2, 0.6, 1.0, 1.4, 1.8]
# width = [1,2,3,4,5,6]
width = [1,2,3,4]

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

def distance_metric(arr_p, arr_r):
	d = 0
	for i in range(len(arr_p)):
		d += arr_p[i] - arr_r[i]
		# if arr_r[i] > arr_p[i]: 
		# 	d -= -10
		# else:
		# 	d += np.sqrt(np.power(arr_p[i]-arr_r[i], 2))
	return d 

def calc_kl(k, w, i, dataset):
	kl_ur_p = []
	kl_r_p = []

	if dataset == 0:
		maxent_ur_up, maxent_r_up = read_up_prob('../output/d'+str(k)+'_expt2.2/syn_maxent_up'+str(i)+'_w'+str(w)+'.pickle')
	else:
		maxent_ur_up, maxent_r_up = read_up_prob('../output_s'+str(dataset)+'/d'+str(k)+'_expt2.2/syn_maxent_up'+str(i)+'_w'+str(w)+'.pickle')
	
	for p in perturb_prob:
		if dataset == 0:
			maxent_file = '../output/d'+str(k)+'_expt2.2/syn_maxent_p'+str(i)+'_p'+str(p)+'_w'+str(w)+'.pickle'
		else:
			maxent_file = '../output_s'+str(dataset)+'/d'+str(k)+'_expt2.2/syn_maxent_p'+str(i)+'_p'+str(p)+'_w'+str(w)+'.pickle'

		maxent_ur_p, maxent_r_p = read_p_prob(maxent_file)

		p = np.array(maxent_ur_up)
		q = np.array(maxent_ur_p)
		r = np.array(maxent_r_up)
		s = np.array(maxent_r_p)
		try:
			kl_1, p_val_1 = power_divergence(f_obs=q, f_exp=p, lambda_="cressie-read")
			kl_2, p_val_2 = power_divergence(f_obs=s, f_exp=r, lambda_="cressie-read")
			# kl_1 = kl_divergence(p, q)
			# kl_2 = kl_divergence(r, s)
		except:
			p[p == 0] = 1e-300
			r[r == 0] = 1e-300
			kl_1, p_val_1 = power_divergence(f_obs=q, f_exp=p, lambda_="cressie-read")
			kl_2, p_val_2 = power_divergence(f_obs=s, f_exp=r, lambda_="cressie-read")

		kl_ur_p.append(kl_1)
		kl_r_p.append(kl_2)

	distance = distance_metric(kl_ur_p, kl_r_p)
	return distance 

def plot(k, dataset=0):
	distances = {}
	for i in range(1,6):
		distances[i] = []
		for w in width: 
			distances[i].append(calc_kl(k, w, i, dataset))

	# plt.figure()
	dist_data = pd.DataFrame(distances, index=[0.002, 0.2, 0.5, 1])
	print("Width vs. Distance Metric: \n", dist_data)
	if dataset!=0:
		dist_data.to_csv('outfiles/rq2.2/ds'+str(dataset)+'_d'+str(k)+'.csv')
	else:
		dist_data.plot(title='Box width performance: '+str(k)+' diseases', grid=True, legend=True, marker='o')
		# plt.style.use('seaborn-darkgrid')
		# plt.legend(fontsize=9)
		plt.title('Box width performance: '+str(k)+' diseases')
		plt.xlabel('Width')
		plt.ylabel('Distance Metric')
		plt.show() 

num_dis = sys.argv[1]
if len(sys.argv) > 2:
	dataset = sys.argv[2]
	plot(int(num_dis), int(dataset))
else:
	plot(int(num_dis))