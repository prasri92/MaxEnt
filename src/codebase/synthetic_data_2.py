#Parallelizing code 
import data_helper
import data_generator 
from multiprocessing import Process
import numpy as np

#Define global parameters 
num_diseases = 20
clusters = 5
# uniform distribution
expon_lambda = 1.0
zipfian_parameter = 1.0

x = np.arange(1, clusters+1)
tau = x**(-zipfian_parameter)
tau = tau/sum(tau)
print(tau)

beta= [0.2, 0.2, 0.2, 0.2, 0.2]
p = 0.5

file_name_real = '../../output/top20diseases_real_expt18.pickle'

dataset_size = 30000
file_name_synthetic = "../../dataset/synthetic_data_expt18.csv"


def get_true_distribution():
	print(data_helper.run(file_name_real, num_diseases, clusters, tau, beta, p))
	print("Data Helper done!")

def get_synthetic_data():
	data_generator.run(file_name_synthetic, dataset_size, expon_lambda, zipfian_parameter, num_diseases, clusters, tau, beta, p)
	print("Data Generated successfully!")

def main():
	p1 = Process(target=get_true_distribution)
	p2 = Process(target=get_synthetic_data)
	p1.start()
	p2.start()

if __name__ == '__main__':
	main()