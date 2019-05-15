#PYTHON3 
import numpy as np
import pickle 

def read_prob_dist(filename):
	with open(filename, "rb") as outfile:
		prob = pickle.load(outfile,encoding='latin1')
	return prob

def kl_divergence(p, q):
	return (p*np.log(p/q)).sum()


def main():
	actual = '../output/top20diseases_actual.pickle'
	synthetic = '../output/top20diseases_synthetic.pickle'
	p = read_prob_dist(actual)
	q = read_prob_dist(synthetic)
	p = np.array(p)
	q = np.array(q)
	kl = kl_divergence(q,p)
	print(kl)

if __name__ == '__main__':
	main()