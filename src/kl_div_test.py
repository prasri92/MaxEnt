#PYTHON3 
import numpy as np
import pickle 
import matplotlib.pyplot as plt

def read_prob_dist(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[0]

def read_prob_sum(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[1], prob[2]

def kl_divergence(p, q):
	return (p*np.log(p/q)).sum()


def main(i):
	actual = '../output/top20diseases_real_expt'+str(i)+'.pickle'
	synthetic = '../output/syn_maxent_expt'+str(i)+'.pickle'
	p = (np.array(read_prob_dist(actual)))
	q = (np.array(read_prob_dist(synthetic)))
	maxent_prob, emp_prob = read_prob_sum(synthetic)
	kl_div = kl_divergence(q,p)
	# print(kl_div)
	num_feats = 20
	xvec = [i+1 for i in range(num_feats + 1)]
	x_ticks = np.arange(0, num_feats+2, 1.0)
	plot_lims = [0,  num_feats+2, -0.1, 1.0]

	plt.plot(xvec, maxent_prob, 'r', label='maxent')
	plt.plot(xvec, emp_prob, 'b', label='empirical')
	plt.axis(plot_lims)
	plt.xticks(x_ticks)
	plt.title('KL div = ' + str(kl_div) + '\nLambda = 1.25, Skew = 2.0')
	plt.legend()
	plt.show()
	# plt.savefig('../figures/'+str(i)+'.png')

if __name__ == '__main__':
	main(25)