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
	actual = '../output/d5000/truedist_expt'+str(i)+'.pickle'
	synthetic = '../output/d5000/syn_maxent_expt'+str(i)+'.pickle'
	p = (np.array(read_prob_dist(actual)))
	q = (np.array(read_prob_dist(synthetic)))
	maxent_prob, emp_prob = read_prob_sum(synthetic)
	kl_div = kl_divergence(q,p)

	return maxent_prob, emp_prob, kl_div

def main_2(i):
	actual = '../output/d5000/truedist_expt'+str(i)+'.pickle'
	synthetic = '../output/d5000/syn_maxent_expt'+str(i)+'_s0.001.pickle'
	p = (np.array(read_prob_dist(actual)))
	q = (np.array(read_prob_dist(synthetic)))
	maxent_prob, emp_prob = read_prob_sum(synthetic)
	kl_div = kl_divergence(q,p)

	return maxent_prob, emp_prob, kl_div

if __name__ == '__main__':
	maxent_prob = []
	emp_prob = []
	kl = []
	for i in range(1,6):
		m, e, k = main_1(i)
		maxent_prob.append(m)
		emp_prob.append(e)
		kl.append(k)

	for i in range(1, 6):
		m2, e2, k2 = main_2(i)
		maxent_prob.append(m2)
		emp_prob.append(e2)
		kl.append(k2)

	num_feats = 20
	xvec = [i+1 for i in range(num_feats + 1)]
	x_ticks = np.arange(0, num_feats+2, 1.0)
	plot_lims = [0,  num_feats+2, -0.1, 1.0]

	fig, ((ax1,ax2,ax3,ax4,ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2,5, figsize=(28, 5), sharey=True)

	ax1.plot(xvec, maxent_prob[0], 'r', label='maxent')
	ax1.plot(xvec, emp_prob[0], 'b', label='empirical')
	ax1.axis(plot_lims)
	ax1.set_xticks(x_ticks)
	ax1.set_title('KL div = ' + str(kl[0]) + '\nLambda = 0.8, Skew = 0.0')
	ax1.legend()
	
	ax2.plot(xvec, maxent_prob[1], 'r', label='maxent')
	ax2.plot(xvec, emp_prob[1], 'b', label='empirical')
	ax2.axis(plot_lims)
	ax2.set_xticks(x_ticks)
	ax2.set_title('KL div = ' + str(kl[1]) + '\nLambda = 0.8, Skew = 1.0')
	ax2.legend()

	ax3.plot(xvec, maxent_prob[2], 'r', label='maxent')
	ax3.plot(xvec, emp_prob[2], 'b', label='empirical')
	ax3.axis(plot_lims)
	ax3.set_xticks(x_ticks)
	ax3.set_title('KL div = ' + str(kl[2]) + '\nLambda = 0.8, Skew = 2.0')
	ax3.legend()

	ax4.plot(xvec, maxent_prob[3], 'r', label='maxent')
	ax4.plot(xvec, emp_prob[3], 'b', label='empirical')
	ax4.axis(plot_lims)
	ax4.set_xticks(x_ticks)
	ax4.set_title('KL div = ' + str(kl[3]) + '\nLambda = 0.8, Skew = 3.0')
	ax4.legend()

	ax5.plot(xvec, maxent_prob[4], 'r', label='maxent')
	ax5.plot(xvec, emp_prob[4], 'b', label='empirical')
	ax5.axis(plot_lims)
	ax5.set_xticks(x_ticks)
	ax5.set_title('KL div = ' + str(kl[4]) + '\nLambda = 0.8, Skew = 4.0')
	ax5.legend()

	plt.figtext(0.5, 0.95, "Support = 0.002")

	ax6.plot(xvec, maxent_prob[5], 'r', label='maxent')
	ax6.plot(xvec, emp_prob[5], 'b', label='empirical')
	ax6.axis(plot_lims)
	ax6.set_xticks(x_ticks)
	ax6.set_title('KL div = ' + str(kl[5]) + '\nLambda = 0.8, Skew = 0.0')
	ax6.legend()
	
	ax7.plot(xvec, maxent_prob[6], 'r', label='maxent')
	ax7.plot(xvec, emp_prob[6], 'b', label='empirical')
	ax7.axis(plot_lims)
	ax7.set_xticks(x_ticks)
	ax7.set_title('KL div = ' + str(kl[6]) + '\nLambda = 0.8, Skew = 1.0')
	ax7.legend()

	ax8.plot(xvec, maxent_prob[7], 'r', label='maxent')
	ax8.plot(xvec, emp_prob[7], 'b', label='empirical')
	ax8.axis(plot_lims)
	ax8.set_xticks(x_ticks)
	ax8.set_title('KL div = ' + str(kl[7]) + '\nLambda = 0.8, Skew = 2.0')
	ax8.legend()

	ax9.plot(xvec, maxent_prob[8], 'r', label='maxent')
	ax9.plot(xvec, emp_prob[8], 'b', label='empirical')
	ax9.axis(plot_lims)
	ax9.set_xticks(x_ticks)
	ax9.set_title('KL div = ' + str(kl[8]) + '\nLambda = 0.8, Skew = 3.0')
	ax9.legend()

	ax10.plot(xvec, maxent_prob[9], 'r', label='maxent')
	ax10.plot(xvec, emp_prob[9], 'b', label='empirical')
	ax10.axis(plot_lims)
	ax10.set_xticks(x_ticks)
	ax10.set_title('KL div = ' + str(kl[9]) + '\nLambda = 0.8, Skew = 4.0')
	ax10.legend()

	plt.figtext(0.5, 0.5, 'Support = 0.001')
	plt.subplots_adjust(hspace = 0.7)
	plt.show()
	# plt.savefig('../figures/'+'Lambda=0.8, support=0.002'+'.png')