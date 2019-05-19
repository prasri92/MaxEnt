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

def main():
	file = '../output/realdata_maxent.pickle'
	num_feats = 20
	xvec = [i+1 for i in range(num_feats + 1)]
	x_ticks = np.arange(0, num_feats+2, 1.0)
	plot_lims = [0,  num_feats+2, -0.1, 1.0]

	maxent_prob, emp_prob = read_prob_sum(file)

	plt.plot(xvec, maxent_prob, 'r', label='maxent')
	plt.plot(xvec, emp_prob, 'b', label='empirical')
	plt.axis(plot_lims)
	plt.xticks(x_ticks)
	plt.title('Maxent vs. empirical probability for patient data')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()