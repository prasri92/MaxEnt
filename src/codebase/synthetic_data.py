import data_helper
import data_generator 
from multiprocessing import Process
import numpy as np
import sys
import configparser


class Synthetic_object(object):
	'''
	Class to hold parameters for synthetic data generation
	'''
	def __init__(self, section):
		config = configparser.ConfigParser()
		config.read('synthetic_data.ini')

		#Get parameters from config file
		self.num_diseases = int(config[section]['num_diseases'])
		self.clusters = int(config[section]['clusters'])
		self.size = int(config[section]['size'])
		self.p = float(config[section]['p'])
		self.q1 = float(config[section]['q1'])
		self.q2 = float(config[section]['q2'])
		#set constant for zipfian parameter
		self.expon_parameter = [0.8,1.2,1.6,2.0,2.4]
		self.z = 2.0
		if self.clusters == 2:
			self.beta = [0.5, 0.5]
		elif self.clusters == 3:
			self.beta = [0.3,0.4,0.3]
		
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
			file_name_real = '../../output/'+'d'+str(self.num_diseases)+'/truedist_expt'+str(file_num)+'_2.pickle'
			file_name_synthetic = "../../dataset/"+"d"+str(self.num_diseases)+"/synthetic_data_expt"+str(file_num)+"_2.csv"
			p1 = Process(target=self.get_true_distribution, args=(file_name_real, tau, l))
			p2 = Process(target=self.get_synthetic_data, args=(file_name_synthetic, l, self.z, self.beta, self.size))
			p1.start()
			p2.start()
			p1.join()
			p2.join()

if __name__ == '__main__':
	# pass section name in the sys argv
	section_name = sys.argv[1]
	obj = Synthetic_object(section_name)
	obj.main()