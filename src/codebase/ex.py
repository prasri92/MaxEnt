import pandas as pd 
import itertools
import numpy as np

def build_constraint_matrix(num_feats):
	A_eq = np.zeros([2**num_feats, 2**num_feats])
	d = list(itertools.product([0,1], repeat=num_feats))
	
	i,j=0,0
	for dis in d:
		index_d = [i for i,val in enumerate(dis) if val==1]
		r = list(itertools.product([0,1], repeat=num_feats))
		j=0
		for rvec in r:
			index_r = [i for i,val in enumerate(rvec) if val==1]
			if(all(x in index_d for x in index_r)):
				A_eq[j][i]=1
			j+=1
		i+=1
	A_eq[0] = np.zeros([2**num_feats])
	A_eq[0][0]=1
	print(A_eq)


build_constraint_matrix(2)

"""
A_eq = np.array([[1, 0, 0, 0],[0, 0, 1, 0],[0, 1, 0, 0],[0, 0, 0, 1]])
print(A_eq)
directory = '../../dataset/d25_2/synthetic_data_expt'+str(1)+'.csv'
data=pd.read_csv(directory, error_bad_lines=False)
a = len(data.values)

counts = {}
options = itertools.product([0,1], repeat=2)
for opt in options:
	counts[opt] = 0

for item in data.values:
	for key in counts.keys():
		if np.array_equal(item, key):
			counts[key]+=1

b_eq = [x/a for x in counts.values()]
print(b_eq)

options = itertools.product([0,1], repeat=2)
f = np.array([x for x in options])
print(f.shape)
"""