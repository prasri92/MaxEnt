import data_helper
import data_generator 
from multiprocessing import Process
import numpy as np

#Define global parameters 
num_diseases = 20
clusters = 5

expon_parameter = [0.8, 1.6, 2.4, 3.2, 4.0]
zipfian_parameter = [0.0, 1.0, 2.0, 3.0, 4.0]

#Change for each expt 
dataset = [500, 1000, 3000]

def generate_tau(z):
	x = np.arange(1, clusters+1)
	tau = x**(-z)
	tau = tau/sum(tau)
	return tau

#check what kind of distribution it should be - as of now uniform
beta = [0.2, 0.2, 0.2, 0.2, 0.2]
p = 0.5
q1 = 0.5
q2 = 0.5

def get_true_distribution(file_name_real, tau, l):
	print(data_helper.run(file_name_real, num_diseases, clusters, l, tau, beta, p))
	print("Data Helper done!")

def get_synthetic_data(file_name_synthetic, expon_parameter, zipfian_parameter, beta, dataset_size):
	data_generator.run(file_name_synthetic, dataset_size, expon_parameter, zipfian_parameter, num_diseases, clusters, beta, p)
	print("Data Generated successfully!")

def main():	
	for size in dataset:
		file_num = 1
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

if __name__ == '__main__':
	main()