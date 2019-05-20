import data_helper
import data_generator 
from multiprocessing import Process
import numpy as np

#Define global parameters 
num_diseases = 20
clusters = 5
# uniform distribution
#test skew 
expon_lambda=[1.6]
# expon_lambda = [0.8, 1.6, 2.4, 3.2, 4.0]
zipfian_parameter = [0.0, 1.0, 2.0, 3.0, 4.0]

#Change for each expt 
dataset = [5000, 10000, 15000, 20000]
dataset_size = dataset[0]

def generate_tau(z):
	x = np.arange(1, clusters+1)
	tau = x**(-z)
	tau = tau/sum(tau)
	return tau

beta = [0.2, 0.2, 0.2, 0.2, 0.2]
p = 0.5

def get_true_distribution(file_name_real, tau):
	print(data_helper.run(file_name_real, num_diseases, clusters, tau, beta, p))
	print("Data Helper done!")

def get_synthetic_data(file_name_synthetic, expon_lambda, zipfian_parameter, tau):
	data_generator.run(file_name_synthetic, dataset_size, expon_lambda, zipfian_parameter, num_diseases, clusters, tau, beta, p)
	print("Data Generated successfully!")

def main():	
	file_num = 6
	for l in expon_lambda:
		for z in zipfian_parameter:
			tau = generate_tau(z)
			file_name_real = '../../output/d5000/truedist_expt'+str(file_num)+'.pickle'
			file_name_synthetic = "../../dataset/d5000/synthetic_data_expt"+str(file_num)+".csv"

			p1 = Process(target=get_true_distribution, args=(file_name_real, tau))
			p2 = Process(target=get_synthetic_data, args=(file_name_synthetic, l, z, tau))
			p1.start()
			p2.start()
			p1.join()
			p2.join()
			file_num+=1

if __name__ == '__main__':
	main()