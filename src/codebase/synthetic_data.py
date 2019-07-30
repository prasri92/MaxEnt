import data_helper
import data_generator 
from multiprocessing import Process
import numpy as np
import sys

#Define global parameters 
num_diseases = 4
clusters = 2

expon_parameter = [0.8, 1.6, 2.4, 3.2, 4.0]
zipfian_parameter = [0.0, 1.0, 2.0, 3.0, 4.0]

#Change for each expt 
size = 50

def generate_tau(z):
	x = np.arange(1, clusters+1)
	tau = x**(-z)
	tau = tau/sum(tau)
	return tau

#check what kind of distribution it should be? 
beta = [0.6, 0.4]
p = 0.6
q1 = 0.3
q2 = 0.7

def get_true_distribution(file_name_real, tau, l):
	print(data_helper.run(file_name_real, num_diseases, clusters, l, tau, beta, p, q1, q2))
	print("Data Helper done!")

def get_synthetic_data(file_name_synthetic, expon_parameter, zipfian_parameter, beta, dataset_size):
	data_generator.run(file_name_synthetic, dataset_size, expon_parameter, zipfian_parameter, num_diseases, clusters, beta, p, q1, q2, overlap=True)
	print("Data Generated successfully!")

'''
def main():	
	for size in dataset:
		file_num = 11
		for l in expon_parameter:
			for z in zipfian_parameter:
				tau = generate_tau(z)
				file_name_real = '../../output/'+'d'+str(size)+'/truedist_expt'+str(file_num)+'.pickle'
				file_name_synthetic = "../../dataset/"+"d"+str(size)+"/synthetic_data_expt"+str(file_num)+".csv"

				p1 = Process(target=get_true_distribution, args=(file_name_real, tau, l))
				p2 = Process(target=get_synthetic_data, args=(file_name_synthetic, l, z, beta, size))
				p1.start()
				p2.start()
				p1.join()
				p2.join()
				file_num+=1
'''
def main(l, z, file_num):
	tau = generate_tau(z)
	file_name_real = '../../output/'+'d'+str(size)+'_4/truedist_expt'+str(file_num)+'.pickle'
	file_name_synthetic = "../../dataset/"+"d"+str(size)+"_4/synthetic_data_expt"+str(file_num)+".csv"
	p1 = Process(target=get_true_distribution, args=(file_name_real, tau, l))
	p2 = Process(target=get_synthetic_data, args=(file_name_synthetic, l, z, beta, size))
	p1.start()
	p2.start()
	p1.join()
	p2.join()

if __name__ == '__main__':
	l,z,file_num = sys.argv[1], sys.argv[2], sys.argv[3]
	main(float(l), float(z), int(file_num))