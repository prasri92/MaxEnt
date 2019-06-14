'''
Python = 3.7 
matplotlib to plot figures
'''
#PYTHON3 
import numpy as np
import pickle 
import matplotlib.pyplot as plt
import csv

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
	actual = '../output/d500_overlap/truedist_expt'+str(i)+'.pickle'
	synthetic = '../output/d500_overlap_nonzeros/syn_maxent_expt'+str(i)+'.pickle' #'_support_0.4.pickle'
	p = (np.array(read_prob_dist(actual)))
	q = (np.array(read_prob_dist(synthetic)))
	maxent_prob, emp_prob = read_prob_sum(synthetic)
	kl_div = kl_divergence(q,p)

	return maxent_prob, emp_prob, kl_div

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
	xvec = [i for i in range(num_feats+1)]
	x_ticks = np.arange(-1, num_feats+1, 1.0)
	plot_lims = [-1,  num_feats+1, -0.1, 1.0]


	'''
	fig, (ax0,ax1,ax2,ax3,ax4) = plt.subplots(1,5, figsize=(30,10))
	lst = [ax0, ax1, ax2, ax3, ax4]

	for num,i in enumerate(lst):
		print(maxent_prob[num])
		i.plot(xvec, maxent_prob[num], 'r', label='Maxent')
		i.plot(xvec, emp_prob[num], 'b', label='Empirical')
		i.axis(plot_lims)
		i.set_xticks(x_ticks)
		# for xy in zip(xvec, maxent_prob[num]):
		# 	i.annotate(('%s, %s') %xy, xy=xy, textcoords='offset points')
		if num//5 == 0:
			i.set_title('KL div = ' + str(kl[num]) + '\nLambda = 1.25, Skew = ' + str(num%5))
		i.legend()

	'''
	fig, ((ax0,ax1,ax2,ax3,ax4), (ax5, ax6, ax7, ax8, ax9), \
		(ax10,ax11, ax12, ax13, ax14 ), (ax15,ax16, ax17, ax18, ax19 ), \
		(ax20,ax21, ax22, ax23, ax24)) = plt.subplots(5,5, figsize=(100, 50))
	lst = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, \
	ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22, ax23, ax24]
	
	# fig, ((ax0,ax1,ax2,ax3,ax4), (ax5, ax6, ax7, ax8, ax9), \
	# 	(ax10,ax11, ax12, ax13, ax14)) = plt.subplots(3,5, figsize=(100, 50))
	# lst = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14]

	data = ["File Number"]
	for i in range(5):
		data.append(i)

	out = csv.writer(open("../output/prob_dist/nz_20d_s500.csv","w"), delimiter=',',quoting=csv.QUOTE_ALL)
	out.writerow(data)
	
	print(len(maxent_prob))
	for num,i in enumerate(lst):
		row = [num+1]
		for prob in maxent_prob[num]:
			row.append(prob)
		out.writerow(row)
		
		i.plot(xvec, maxent_prob[num], 'r', label='Maxent')
		i.plot(xvec, emp_prob[num], 'b', label='Empirical')
		i.axis(plot_lims)
		i.set_xticks(x_ticks)
		# for xy in zip(xvec, maxent_prob[num]):
		# 	i.annotate(('%s, %s') %xy, xy=xy, textcoords='offset points')
		if num//5 == 0:
			i.set_title('KL div = ' + str(kl[num]) + '\nLambda = 1.25, Skew = ' + str(num%5))
		elif num//5 == 1:
			i.set_title('KL div = ' + str(kl[num]) + '\nLambda = 0.625, Skew = ' + str(num%5))
		elif num//5 == 2:
			i.set_title('KL div = ' + str(kl[num]) + '\nLambda = 0.416, Skew = ' + str(num%5))
		elif num//5 == 3:
			i.set_title('KL div = ' + str(kl[num]) + '\nLambda = 0.3125, Skew = ' + str(num%5))
		elif num//5 == 4:
			i.set_title('KL div = ' + str(kl[num]) + '\nLambda = 0.25, Skew = ' + str(num%5))
		i.legend()
		row = []

	fig.suptitle('Diseases = 20\nDataset Size = 500', fontsize=30)
	plt.subplots_adjust(hspace = 0.7)
	plt.show()