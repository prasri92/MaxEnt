import data_helper
import data_generator 
from multiprocessing import Process
import numpy as np
import sys
import random
import csv

'''
Generate data for multiple configurations, through a random grid search

Store the choices in a data file regarding n, exponent and size, rest of the parameters are chosen 
from a uniform distribution with bounds 
'''
class Synthetic_object(object):
	'''
	Class to hold parameters for synthetic data generation
	'''
	def __init__(self, n, ds_num, f_num):
		'''
		opt_dict will have the following parameters:
			n: number of diseases 
			f: file number for storage 
		'''
		print('DATASET NUMBER: ', ds_num, ' DISEASES: ', n, ' FILE NUMBER: ', f_num)
		parameter_file = '../../parameters/dataset_s'+str(ds_num)+'_d'+str(n)+'.csv'
		with open(parameter_file, 'r') as f:
			lines=f.readlines()
			params = lines[int(f_num)]
		f.close()

		params = params.split(',')

		self.num_diseases = int(params[0])
		self.clusters = int(params[1]) 
		self.size = int(params[2])
		self.p = float(params[3])
		self.q1 = float(params[4])
		self.z = float(params[5])
		self.q2 = 0.9

		self.expon_parameter = float(params[7])
		self.beta = [1/self.clusters]*self.clusters
		
		self.dataset_num = int(ds_num)
		self.file_num = int(f_num)

	def generate_tau(self):
		x = np.arange(1, self.clusters+1)
		tau = x**(-self.z)
		tau = tau/sum(tau)
		return tau

	def get_true_distribution(self, file_name_real, tau, l):
		print(data_helper.run(file_name_real, self.num_diseases, self.clusters, l, tau, self.beta, self.p, self.q1, self.q2))
		print("Data Helper done!")

	def get_synthetic_data(self, file_name_synthetic, expon_parameter, zipfian_parameter, beta, dataset_size):
		data_generator.run(file_name_synthetic, dataset_size, expon_parameter, zipfian_parameter, self.num_diseases, self.clusters, beta, self.p, self.q1, self.q2, overlap=True)
		print("Data Generated successfully!")

	def main(self):
		tau = self.generate_tau()
		file_name_real = '../../output/output_s'+str(self.dataset_num)+'/d'+str(self.num_diseases)+'/truedist_expt'+str(self.file_num)+'.pickle'
		file_name_synthetic = "../../data/dataset_s"+str(self.dataset_num)+"/d"+str(self.num_diseases)+"/synthetic_data_expt"+str(self.file_num)+".csv"
		p1 = Process(target=self.get_true_distribution, args=(file_name_real, tau, self.expon_parameter))
		p2 = Process(target=self.get_synthetic_data, args=(file_name_synthetic, self.expon_parameter, self.z, self.beta, self.size))
		p1.start()
		p2.start()
		p1.join()
		p2.join()

if __name__ == '__main__':
	# pass all options in the sys argv
	n = sys.argv[1]
	ds_num = sys.argv[2]
	f_num = sys.argv[3]
	obj = Synthetic_object(n, ds_num, f_num)
	obj.main()
