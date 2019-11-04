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
	# actual = '../output/d50_4/truedist_expt'+str(i)+'.pickle'
	# actual = '../output/d200_4/truedist_expt'+str(i)+'.pickle'
	# actual = '../output/d250_10/truedist_expt'+str(i)+'.pickle'
	# actual = '../output/d350_15/truedist_expt'+str(i)+'.pickle'
	actual = '../output/d500_20/truedist_expt'+str(i)+'.pickle'


	# synthetic = '../output/d50_4/syn_maxent_expt'+str(i)+'.pickle'
	# synthetic = '../output/d200_4/syn_maxent_expt'+str(i)+'.pickle'
	# synthetic = '../output/d250_10/syn_maxent_expt'+str(i)+'.pickle'
	# synthetic = '../output/d350_15/syn_maxent_expt'+str(i)+'.pickle'
	synthetic = '../output/d500_20_mu15/syn_maxent_expt'+str(i)+'.pickle'

	p = (np.array(read_prob_dist(actual)))
	q = (np.array(read_prob_dist(synthetic)))
	maxent_prob, emp_prob = read_prob_sum(synthetic)
	kl_div = kl_divergence(q,p)
	true_prob = read_true_prob(actual)
	return maxent_prob, emp_prob, kl_div, true_prob

if __name__ == '__main__':
	maxent_prob = []
	emp_prob = []
	kl = []
	true_prob = []

	for i in range(3, 24, 10):
		m, e, k, t = compute_prob(i)
		maxent_prob.append(m)
		emp_prob.append(e)
		kl.append(k)
		true_prob.append(t)

	# num_feats = 4
	# num_feats = 10
	# num_feats = 15
	num_feats = 20
	xvec = [i for i in range(num_feats+1)]
	x_ticks = np.arange(0, num_feats+1, 1.0)
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
	# fig, ((ax0,ax1,ax2,ax3,ax4), (ax5, ax6, ax7, ax8, ax9), \
	# 	(ax10,ax11, ax12, ax13, ax14 ), (ax15,ax16, ax17, ax18, ax19 ), \
	# 	(ax20,ax21, ax22, ax23, ax24)) = plt.subplots(5,5, figsize=(40, 20))
	# lst = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, \
	# ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22, ax23, ax24]
	

	# fig, ((ax0,ax1,ax2,ax3,ax4, ax5), (ax6, ax7, ax8, ax9, ax10, ax11), \
	# 	(ax12,ax13, ax14, ax15, ax16, ax17), (ax18,ax19, ax20, ax21, ax22, ax23), \
	# 	(ax24,ax25, ax26, ax27, ax28, ax29)) = plt.subplots(5,6, figsize=(100, 50))
	# lst = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, \
	# ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22, ax23, ax24, ax25, ax26, ax27, ax28, ax29]

	# fig, ((ax0,ax1,ax2,ax3,ax4), (ax5, ax6, ax7, ax8, ax9), \
	# 	(ax10,ax11, ax12, ax13, ax14)) = plt.subplots(3,5, figsize=(40, 20))
	# lst = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14]

	fig, (ax0, ax1, ax2) = plt.subplots(1,3, figsize=(15,4))
	lst = [ax0, ax1, ax2]

	start_file_num = 3
	for num,i in enumerate(lst):
		emp_prob[num] = np.around(emp_prob[num], decimals=4)
		maxent_prob[num] = np.around(maxent_prob[num], decimals=4)
		true_prob[num] = np.around(true_prob[num], decimals=4)

		data = {'Empirical':emp_prob[num], 'Maximum Entropy Prob.':maxent_prob[num], 'True Prob.':true_prob[num]}
		df = pd.DataFrame(data=data)
		# df.to_csv('../output/output_dist/4d_size200_'+str(start_file_num)+'.csv')
		# df.to_csv('../output/output_dist/10d_size250_'+str(start_file_num)+'.csv')
		# df.to_csv('../output/output_dist/15d_size350_'+str(start_file_num)+'.csv')
		# df.to_csv('../output/output_dist/20d_size500_mu10_'+str(start_file_num)+'.csv')
		df.to_csv('../output/output_dist/20d_size500_mu15_'+str(start_file_num)+'.csv')
		
		i.plot(xvec, maxent_prob[num], 'r', label='Maxent')
		i.plot(xvec, emp_prob[num], 'b', label='Empirical')
		# i.plot(xvec, true_prob[num], 'g', label='True')
		i.axis(plot_lims)
		i.set_xticks(x_ticks)
		# i.xlabel("Number of diseases per patient")
		# i.ylabel("Prob.")
		# for xy in zip(xvec, maxent_prob[num]):
		# 	i.annotate(('%s, %s') %xy, xy=xy, textcoords='offset points')
		if num==0:
			i.set_title('KL div = ' + str(kl[num]) + '\nLambda = 1.25, Skew = 2', fontsize=7)
		elif num==1:
			i.set_title('KL div = ' + str(kl[num]) + '\nLambda = 0.416, Skew = 2', fontsize=7)
		elif num==2:
			i.set_title('KL div = ' + str(kl[num]) + '\nLambda = 0.25, Skew = 2', fontsize=7)

		'''
		if num//5 == 0:
			i.set_title('KL div = ' + str(kl[num]) + '\nLambda = 1.25, Skew = ' + str(num%5), fontsize=7)
		elif num//5 == 1:
			i.set_title('KL div = ' + str(kl[num]) + '\nLambda = 0.625, Skew = ' + str(num%5), fontsize=7)
		elif num//5 == 2:
			i.set_title('KL div = ' + str(kl[num]) + '\nLambda = 0.416, Skew = ' + str(num%5), fontsize=7)
		elif num//5 == 3:
			i.set_title('KL div = ' + str(kl[num]) + '\nLambda = 0.3125, Skew = ' + str(num%5), fontsize=7)
		elif num//5 == 4:
			i.set_title('KL div = ' + str(kl[num]) + '\nLambda = 0.25, Skew = '+ str(num%5), fontsize=7)
		'''
		i.legend(fontsize=6)
		row = []
		start_file_num += 10

	# fig.suptitle('Diseases = 4, Dataset Size = 200\n', y=0.99, fontsize=10)
	# fig.suptitle('Diseases = 10, Dataset Size = 250\n', y=0.99, fontsize=10)
	# fig.suptitle('Diseases = 15, Dataset Size = 350\n', y=0.99, fontsize=10)
	fig.suptitle('Diseases = 20, Dataset Size = 500, Maximum cluster size = 15\n', y=0.99, fontsize=10)
	# plt.subplots_adjust(hspace = 0.6, top=0.85)
	plt.subplots_adjust(top=0.85)
	plt.show()