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

np.seterr(all='raise')
perturb_prob = [0.01, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21]
width = [0.2, 0.6, 1.0, 1.4, 1.8]

# Read the true prob. distribution for sum of diseases
def read_true_prob(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[0], prob[1]

def read_perturb_prob(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[0], prob[1], prob[2], prob[3]

# Get kl divergence of probability distributions
def kl_divergence(p, q):
	return (p*np.log(p/q)).sum()

def distance_metric(arr_p, arr_r):
	d = 0
	for i in range(len(arr_p)):
		if arr_r[i] > arr_p[i]: 
			d -= 100
		else:
		d += np.sqrt(np.power(arr_p[i]-arr_r[i], 2))
	return d 

def calc_kl(k, w, i=3):
	kl_p = []
	kl_r = []

	true_file = '../output/d'+str(k)+'/truedist_expt'+str(i)+'.pickle'
	true_dist, true_prob = read_true_prob(true_file)

	for p in perturb_prob:
		maxent_file = '../output/d'+str(k)+'/syn_maxent_perturb_box_expt'+str(i)+'_p'+str(p)+'_w'+str(w)+'.pickle'
		maxent_p_dist, maxent_p_prob, maxent_r_dist, maxent_r_prob = read_perturb_prob(maxent_file)

		p = np.array(true_dist)
		q = np.array(maxent_p_dist)
		r = np.array(maxent_r_dist)
		try:
			kl_1 = kl_divergence(p, q)
			kl_2 = kl_divergence(p, r)
		except FloatingPointError as e:
			print('Infinity')

		kl_p.append(kl_1)
		kl_r.append(kl_2)

	distance = distance_metric(kl_p, kl_r)
	return distance 

def plot(k):
	distances = []
	for w in width: 
		distances.append(calc_kl(k, w))

	x_ticks = np.arange(0, 2, 0.2)
	plt.style.use('seaborn-darkgrid')

	plt.plot(width, distances, marker='o', label=str(k)+' diseases')

	plt.xticks(x_ticks)
	plt.legend(fontsize=9)
	plt.title('Box width performance')
	plt.xlabel('Width')
	plt.ylabel('Distance Metric')
	plt.show()

num_dis = sys.argv[1]
plot(int(num_dis))
