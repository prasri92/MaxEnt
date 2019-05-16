#PYTHON3 
import numpy as np
import pickle 
import matplotlib.pyplot as plt

#define global num_feats, change later 
num_feats = 20

def read_prob_dist(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob[0]

def kl_divergence(p, q):
	return (p*np.log(p/q)).sum()

def plot(p, q):
	xvec = [i+1 for i in range(num_feats + 1)]
	# ~xvec
	x_ticks = np.arange(0, num_feats+2, 1.0)
	# ~x_ticks
	plot_lims = [0,  num_feats+2, -0.1, 1.0]
	# ~plot_lims
	# Both on same plot
	plt.figure()
	plt.plot(xvec, p, 'ro', label='empirical')  # empirical
	plt.plot(xvec, q, 'bo', label='maxent')  # maxent
	plt.legend()
	plt.xticks(x_ticks)
	plt.axis(plot_lims)
	plt.show()
	# plt.savefig('../out/plot_merge_' + str(k_val) + '.png')

def main():
	actual = '../output/top20diseases_real_expt4.pickle'
	synthetic = '../output/syn_maxent_expt4.pickle'
	p = read_prob_dist(actual)
	q = read_prob_dist(synthetic)
	p = np.array(p)
	q = np.array(q)
	plot(p,q)
	kl = kl_divergence(q,p)
	print(kl)

if __name__ == '__main__':
	main()