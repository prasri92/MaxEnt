from scipy.optimize import linprog
import itertools 
import numpy as np

# x_bounds = (0,1)
A_eq = [[1,0,0,0,0,0,0,0],[0,1,0,1,0,1,0,1],[0,0,1,1,0,0,1,1],[0,0,0,1,0,0,0,1],[0,0,0,0,1,1,1,1],[0,0,0,0,0,0,1,1],[0,0,0,0,0,0,0,1],[1,1,1,1,1,1,1,1]]
b_eq = [0.9514285714285714, 0.0142857142857143, 0.0257142857142857, 0.00571428571428571, 0.02, 0.00571428571428571, 0, 1]

num_feats = 3
# d = list(itertools.product([0,1], repeat=num_feats))

x = [0,0,0,0,0,-1,0,-1]

A_ub = np.identity(2**num_feats)
b_ub = np.ones(2**num_feats)

res = linprog(x, A_eq=A_eq, b_eq=b_eq, options={'disp':True}) #A_ub=A_ub, b_ub=b_ub, options={"disp": True})
print(res)
res = {'message': res.message, 'status':res.status, 'x': res.x if res.success else None}
print(res['message'])

zero_vectors = []
for vector, lp_prob in enumerate(res['x']):
    if lp_prob == 0:
        flag = 1
        #which vector it is will have to be calculated differently
        # if vector not in remove_indices:
        zero_vectors.append(vector)

zero_vectors = [format(x, '0'+str(num_feats)+'b') for x in zero_vectors]
print("Zero vectors are: ", zero_vectors)