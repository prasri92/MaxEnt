'''
Python = 3.7 
matplotlib to plot figures
'''
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


def main_1(i):
	actual = '../output/d500/truedist_expt'+str(i)+'.pickle'
	synthetic = '../output/d500/syn_maxent_expt'+str(i)+'.pickle'
	p = (np.array(read_prob_dist(actual)))
	q = (np.array(read_prob_dist(synthetic)))
	maxent_prob, emp_prob = read_prob_sum(synthetic)
	kl_div = kl_divergence(p,q)

	return maxent_prob, emp_prob, kl_div
'''
#TO compare support values 
def main_2(i):
	actual = '../output/d5000/truedist_expt'+str(i)+'.pickle'
	synthetic = '../output/d5000/syn_maxent_expt'+str(i)+'_s0.001.pickle'
	p = (np.array(read_prob_dist(actual)))
	q = (np.array(read_prob_dist(synthetic)))
	maxent_prob, emp_prob = read_prob_sum(synthetic)
	kl_div = kl_divergence(q,p)

	return maxent_prob, emp_prob, kl_div
'''
if __name__ == '__main__':
	maxent_prob = []
	emp_prob = []
	kl = []
	for i in range(1,26):
		m, e, k = main_1(i)
		maxent_prob.append(m)
		emp_prob.append(e)
		kl.append(k)

	num_feats = 20
	xvec = [i+1 for i in range(num_feats + 1)]
	x_ticks = np.arange(0, num_feats+2, 1.0)
	plot_lims = [0,  num_feats+2, -0.1, 1.0]


	fig, ((ax0,ax1,ax2,ax3,ax4), (ax5, ax6, ax7, ax8, ax9), \
		(ax10,ax11, ax12, ax13, ax14 ), (ax15,ax16, ax17, ax18, ax19 ), \
		(ax20,ax21, ax22, ax23, ax24)) = plt.subplots(5,5, figsize=(28, 5))
	lst = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, \
	ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22, ax23, ax24]

	for num,i in enumerate(lst):
		i.plot(xvec, maxent_prob[num], 'r', label='maxent')
		i.plot(xvec, emp_prob[num], 'b', label='empirical')
		i.axis(plot_lims)
		i.set_xticks(x_ticks)
		if num//5 == 0:
			i.set_title('KL div = ' + str(kl[num]) + '\nLambda = 0.8, Skew = ' + str(num%5))
		elif num//5 == 1:
			i.set_title('KL div = ' + str(kl[num]) + '\nLambda = 1.6, Skew = ' + str(num%5))
		elif num//5 == 2:
			i.set_title('KL div = ' + str(kl[num]) + '\nLambda = 2.4, Skew = ' + str(num%5))
		elif num//5 == 3:
			i.set_title('KL div = ' + str(kl[num]) + '\nLambda = 3.2, Skew = ' + str(num%5))
		elif num//5 == 4:
			i.set_title('KL div = ' + str(kl[num]) + '\nLambda = 4.0, Skew = ' + str(num%5))
		i.legend()

	plt.subplots_adjust(hspace = 0.7)
	plt.show()