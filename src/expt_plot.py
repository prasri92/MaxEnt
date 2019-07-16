'''
Python = 3.7 
matplotlib to plot figures
'''
#PYTHON3 
import numpy as np
import pickle 
import matplotlib.pyplot as plt
import csv
import sys 

def read_prob_dist(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[0]

def read_prob_sum(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[1], prob[2]

def read_true_prob(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[1]

def kl_divergence(p, q):
	return (p*np.log(p/q)).sum()


def compute_prob(i, j=None):
	actual = '../output/d500_20/truedist_expt'+str(i)+'.pickle'
	synthetic = '../output/d500_20_add01234/syn_maxent_expt'+str(i)+'.pickle'
	p = (np.array(read_prob_dist(actual)))
	q = (np.array(read_prob_dist(synthetic)))
	maxent_prob, emp_prob = read_prob_sum(synthetic)
	kl_div = kl_divergence(q,p)
	true_prob = read_true_prob(actual)
	return maxent_prob, emp_prob, kl_div, true_prob

def main(i=None):
	m, e, k, t = compute_prob(i)

	num_feats = 20
	xvec = [i for i in range(num_feats+1)]
	x_ticks = np.arange(0, num_feats+1, 1.0)
	plot_lims = [-1,  num_feats+1, -0.1, 1.0]
	plt.figure(figsize=(40,40))
	plt.plot(xvec, m, 'r', label='Maxent')
	plt.plot(xvec, e, 'b', label='Empirical')
	# plt.plot(xvec, t, 'g', label='True')
	plt.axis(plot_lims)
	plt.xticks(x_ticks)

	for x,y in enumerate(m):
		plt.text(x,y+0.02,s=str(round(y,2)),fontsize=8)
	plt.legend(fontsize=10)
	if int(i)//5 == 0:
		plt.title('KL div = ' + str(k) + '\nLambda = 1.25, Skew = ' + str(int(i)%5), fontsize=10)
	elif int(i)//5 == 1:
		plt.title('KL div = ' + str(k) + '\nLambda = 0.625, Skew = ' + str(int(i)%5), fontsize=10)
	elif int(i)//5 == 2:
		plt.title('KL div = ' + str(k) + '\nLambda = 0.416, Skew = ' + str(int(i)%5), fontsize=10)
	elif int(i)//5 == 3:
		plt.title('KL div = ' + str(k) + '\nLambda = 0.3125, Skew = ' + str(int(i)%5), fontsize=10)
	elif int(i)//5 == 4:
		plt.title('KL div = ' + str(k) + '\nLambda = 0.25, Skew = '+ str(int(i)%5), fontsize=10)

	# plt.title('Diseases = 20\nDataset Size = 500\n0 1 2 3 4 constraint imposed', fontsize=10)
	plt.show()

if __name__ == '__main__':
	i = sys.argv[1]
	main(i)