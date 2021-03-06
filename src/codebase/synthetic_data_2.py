import data_helper
import data_generator 
from multiprocessing import Process
import numpy as np
import sys
import configparser
import json

'''
Used for multiple configurations 
'''
class Synthetic_object(object):
	'''
	Class to hold parameters for synthetic data generation
	'''
	def __init__(self, opt_dict):
		with open('syn_config.json') as config_file:
			data = json.load(config_file)

		section = opt_dict['section_name']
		if 'file' in opt_dict:
			self.file = opt_dict['file']
		else:
			self.file = None

		config_number = opt_dict['config_number']
		config = data['config'+str(config_number)]
		
		#Get parameters from config file
		self.num_diseases = int(config[section]['num_diseases'])
		self.clusters = int(config[section]['clusters'])
		self.size = int(config[section]['size'])
		self.p = float(config['GLOBAL']['p'])
		self.q1 = float(config['GLOBAL']['q1'])
		self.z = float(config['GLOBAL']['z'])
		self.q2 = 0.9

		# Exponent for the exponential distribution 
		self.expon_parameter = [0.8,1.2,1.6,2.0,2.4]

		# choose the probability that each disease belongs a cluster. 
		# Uniform probability chosen for now. 
		if self.clusters == 2:
			self.beta = [0.5, 0.5]
		elif self.clusters == 3:
			self.beta = [0.33,0.33,0.34]
		elif self.clusters == 4:
			self.beta = [0.25, 0.25, 0.25, 0.25]
		elif self.clusters == 6:
			self.beta = [0.17, 0.16, 0.17, 0.16, 0.17, 0.17]

		
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
		for num, l in enumerate(self.expon_parameter):
			file_num = num+1
			tau = self.generate_tau()
			if self.file==None:
				file_name_real = '../../output/d'+str(self.num_diseases)+'/truedist_expt'+str(file_num)+'.pickle'
				file_name_synthetic = "../../dataset/d"+str(self.num_diseases)+"/synthetic_data_expt"+str(file_num)+".csv"
			else:
				file_name_real = '../../output_s'+str(self.file)+'/d'+str(self.num_diseases)+'/truedist_expt'+str(file_num)+'.pickle'
				file_name_synthetic = "../../dataset_s"+str(self.file)+"/d"+str(self.num_diseases)+"/synthetic_data_expt"+str(file_num)+".csv"
			p1 = Process(target=self.get_true_distribution, args=(file_name_real, tau, l))
			p2 = Process(target=self.get_synthetic_data, args=(file_name_synthetic, l, self.z, self.beta, self.size))
			p1.start()
			p2.start()
			p1.join()
			p2.join()

if __name__ == '__main__':
	# pass all options in the sys argv
	options = sys.argv
	if len(sys.argv) > 3:
		opt_dict = {'section_name':str(sys.argv[1]), 'config_number':sys.argv[2], 'file':int(sys.argv[3])}
	else:
		opt_dict = {'section_name':str(sys.argv[1]), 'config_number':sys.argv[2]}
	obj = Synthetic_object(opt_dict)
	obj.main()
