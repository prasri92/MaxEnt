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
	def __init__(self, opt_dict):
		'''
		opt_dict will have the following parameters:
			n: number of diseases 
			f: file number for storage 
		'''
		self.num_diseases = int(opt_dict['n'])
		self.dataset_num = int(opt_dict['dataset_num'])

		self.clusters = random.choice([int(self.num_diseases*(0.25)), int(self.num_diseases*(0.5)), int(self.num_diseases*(0.75))])
		self.size = random.randint(a=1000, b=60000)

		self.p = random.uniform(a=0.25,b=0.75)
		self.q1 = random.uniform(a=0.25,b=0.75)
		self.z = random.uniform(a=1.0,b=4.0)
		self.q2 = 0.9

		self.expon_parameter = np.random.uniform(low=0.8, high=2.4,size=10)
		self.beta = [1/self.clusters]*self.clusters

		self.parameter_file = '../../parameters/dataset_s'+str(self.dataset_num)+'_d'+str(self.num_diseases)+'.csv'
		with open(self.parameter_file, "w") as csvFile: 
			first_row = ['#_diseases','cluster_size','dataset_size','p','q1','z','local_file_num','exponent','beta']
			csv.writer(csvFile).writerow(first_row)
			for index, e in enumerate(self.expon_parameter):
				row = [self.num_diseases, self.clusters, self.size, self.p, self.q1, self.z, index+1, e, self.beta[0]]
				csv.writer(csvFile).writerow(row)
		csvFile.close()

if __name__ == '__main__':
	# pass all options in the sys argv
	options = sys.argv
	opt_dict = {'n':sys.argv[1], 'dataset_num':sys.argv[2]}
	obj = Synthetic_object(opt_dict)