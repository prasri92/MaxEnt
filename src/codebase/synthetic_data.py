import data_helper
import data_generator 
from multiprocessing import Process

#Define global parameters 
num_diseases = 15
clusters = 5
# uniform distribution
tau = [0.2, 0.2, 0.2, 0.2, 0.2]
beta= [0.2, 0.2, 0.2, 0.2, 0.2]
p = 0.5

file_name_real = '../../output/top20diseases_real_uniform.pickle'

dataset_size = 500
file_name_synthetic = "../../dataset/synthetic_data_uniform.csv"

expon_lambda = 0.25
zipfian_parameter = 1.5

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