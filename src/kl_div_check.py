'''
Python = 3.7 
'''
#PYTHON3 
import numpy as np
import pickle 
import matplotlib.pyplot as plt

def read_prob_dist(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[0]

def kl_divergence(p, q):
	return (p*np.log(p/q)).sum()

def main_1(i, j):
	expt1 = '../output/d500/truedist_expt'+str(i)+'.pickle'
	expt2 = '../output/d500/truedist_expt'+str(j)+'.pickle'
	p = (np.array(read_prob_dist(expt1)))
	q = (np.array(read_prob_dist(expt2)))
	kl_div = kl_divergence(q,p)

	print('KL_divergence is: ', kl_div)

if __name__ == '__main__':
	main_1(1, 25)